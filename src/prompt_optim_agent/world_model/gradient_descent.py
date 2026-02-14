# The code is modified based on Automatic Prompt Optimization with "Gradient Descent" and Beam Search
# https://arxiv.org/abs/2305.03495

import re
from typing import Tuple

import numpy as np

from prompt_optim_agent.tracking import MetricsTracker

from .prompts.gradient_descent_prompts import (
    ascend_gradient_prompt_template,
    ascend_optimize_prompt_template,
    ascend_optimize_prompt_template_single,
    example_template,
    example_without_label_template,
    gradient_prompt_template,
    mix_optimize_prompt_template,
    mix_optimize_prompt_template_single,
    optimize_prompt_template,
    optimize_prompt_template_single,
    summarization_prompt_template,
)
from .prompts.log_prompt_templates import (
    forward_log_template,
    gradient_log_template,
    gradient_summary_template,
    optimize_log_template,
)


class GradientDescent:
    def __init__(
        self,
        task,
        base_model,
        optim_model,
        print_log=True,
        logger=None,
        num_new_prompts=1,
        tracker: MetricsTracker = None,
        **kwargs,
    ):

        self.task = task
        self.base_model = base_model
        self.optim_model = optim_model
        self.logger = logger
        self.print_log = print_log if logger is not None else False
        self.num_new_prompts = num_new_prompts
        self.tracker = tracker or MetricsTracker()
        self.positive_reinforcement_depth = kwargs.get(
            "positive_reinforcement_depth", 1
        )
        self.gradient_sampling = kwargs.get("gradient_sampling", 1)
        assert (
            self.gradient_sampling % 1 == 0 and self.gradient_sampling >= 1
        ), "self.gradient_sampling must be an integer greater than or equal to 1."

        def _select_template(single, multi):
            return single if num_new_prompts == 1 else multi

        self.optimize_prompt_template = _select_template(
            optimize_prompt_template_single, optimize_prompt_template
        )
        self.ascend_optimize_prompt_template = _select_template(
            ascend_optimize_prompt_template_single, ascend_optimize_prompt_template
        )
        self.mix_optimize_prompt_template = _select_template(
            mix_optimize_prompt_template_single, mix_optimize_prompt_template
        )
        self.gradient_prompt_template = gradient_prompt_template
        self.ascend_gradient_prompt_template = ascend_gradient_prompt_template
        self.example_template = example_template

        self._build_forward_prompts_func = task.build_forward_prompts_completion
        self._batch_forward_func = self.base_model.batch_forward_func

    def forward(self, batch, cur_prompt):
        batch_size = len(batch["question"])
        batch_prompts = self._build_forward_prompts_func(batch["question"], cur_prompt)
        responses, logging_dict = self._batch_forward_func(batch_prompts)
        self.tracker.log({f"{key}_base_model": value for key, value in logging_dict.items()})

        if self.logger is not None:
            for p, r in zip(batch_prompts, responses):
                self.logger.info(f"Input:\n{p}")
                self.logger.info(f"Output:\n{r}")

        preds = self.task.batch_clean_responses(responses)

        batch_answers = batch.get("answer", None)
        labels = (
            self.task.clean_labels(batch_answers) if batch_answers is not None else None
        )

        correct = self.task.cal_correct(
            preds=preds, questions=batch["question"], labels=labels, prompt=cur_prompt
        )
        acc = self.task.cal_metric_from_cal_correct_output(correct)

        batch_logs = []
        for i in range(batch_size):
            batch_logs.append(
                {
                    "cur_prompt": cur_prompt,
                    "question": batch["question"][i],
                    "model_input": batch_prompts[i],
                    "gt_answer": (
                        batch["answer"][i] if batch_answers is not None else "<NA>"
                    ),
                    "model_response": responses[i],
                    "label": labels[i] if labels is not None else "<NA>",
                    "pred": preds[i],
                }
            )

        forward_output = {
            "cur_prompt": cur_prompt,
            "correct": correct,
            "examples": batch_logs,
            "acc": acc,
        }

        if self.print_log and self.logger is not None:
            log_str = forward_log_template.format(
                cur_prompt=cur_prompt,
                batch_prompts=batch_prompts,
                responses=responses,
                preds=preds,
                labels=labels,
                correct=forward_output["correct"],
                acc=forward_output["acc"],
            )

            self.logger.info(log_str)
        return forward_output

    def _clean_self_eval_score(self, response):
        return re.findall(r"\d+", response)[-1]

    def _split_error_and_correct_examples(self, forward_output) -> Tuple[str, str]:
        error_examples = []
        correct_examples = []
        count = 0
        for i, example in enumerate(forward_output["examples"]):
            if forward_output["correct"][i] == 0:
                count += 1
                error_examples.append(
                    self.example_template.format(
                        index=count,
                        question=example["model_input"],
                        label=example["label"],
                        response=example["model_response"],
                        prediction=example["pred"],
                    )
                )
            elif forward_output["correct"][i] == 1:
                count += 1
                correct_examples.append(
                    self.example_template.format(
                        index=count,
                        question=example["model_input"],
                        label=example["label"],
                        response=example["model_response"],
                        prediction=example["pred"],
                    )
                )
            else:
                raise ValueError(
                    f"_get_error_examples: invalid correct number {i} {forward_output}."
                )
        error_string = "".join(error_examples)
        correct_string = "".join(correct_examples)
        return error_string, correct_string

    def _split_error_and_correct_examples_multi_metric(
        self, forward_output
    ) -> Tuple[str, str]:
        error_examples = []
        correct_examples = []
        correct_metric_based = np.array(forward_output["correct"]).transpose()
        for i, metric in enumerate(self.task.metrics_definition):
            metric_name = metric["metric_name"]
            metric_desc = metric["metric_desc"]
            metric_instruction = metric["metric_instruction"]
            metric_error_examples = []
            metric_correct_examples = []
            for j, example in enumerate(forward_output["examples"]):
                pred_example = correct_metric_based[i, j]
                if pred_example > -1:
                    # this means the example is not N/A
                    if pred_example == 0:
                        metric_error_examples.append(
                            example_without_label_template.format(
                                index=j,
                                question=example["question"],
                                response=example["model_response"],
                                prediction="NO",
                            )
                        )
                    elif pred_example == 1:
                        metric_correct_examples.append(
                            example_without_label_template.format(
                                index=j,
                                question=example["question"],
                                response=example["model_response"],
                                prediction="YES",
                            )
                        )
                    else:
                        raise ValueError(
                            f"invalid evaluation label for the sample {j}, label={pred_example}"
                        )
            if len(metric_error_examples) > 0:
                error_examples.append(
                    f"<metric_name>\n{metric_name}\n</metric_name>\n"
                    f"<metric_description>\n{metric_desc}\n</metric_description>\n"
                    f"<metric_instruction>\n{metric_instruction}\n</metric_instruction>\n"
                    + "".join(metric_error_examples)
                )
            if len(metric_correct_examples) > 0:
                correct_examples.append(
                    f"<metric_name>\n{metric_name}\n</metric_name>\n"
                    f"<metric_description>\n{metric_desc}\n</metric_description>\n"
                    f"<metric_instruction>\n{metric_instruction}\n</metric_instruction>\n"
                    + "".join(metric_correct_examples)
                )
        error_string = "".join(error_examples)
        correct_string = "".join(correct_examples)
        return error_string, correct_string

    def _build_prompt_trajectory_str(self, prompts):
        prompt_path_str_template = "({index}) {prompt}\n"
        return "".join(
            prompt_path_str_template.format(index=i, prompt=prompt)
            for i, prompt in enumerate(prompts)
        )

    def _get_gradient_summary_prompt(self, batch_gradient):
        feedbacks = "<feedbacks>\n"
        for i, gradient in enumerate(batch_gradient):
            feedbacks += "<feedback {i}>\n{gradient}\n</feedback {i}>\n".format(
                i=i, gradient=gradient
            )
        feedbacks += "</feedbacks>"
        summary_prompt = summarization_prompt_template.format(
            nb_feedbacks=len(batch_gradient), feedbacks=feedbacks
        )
        return summary_prompt

    def cal_gradient(
        self,
        cur_prompt,
        example_string,
        gradient_prompt_template,
        nb_gradient_samples: int = 1,
    ):
        assert (
            nb_gradient_samples >= 1 and nb_gradient_samples % 1 == 0
        ), "nb_gradient_samples must be an integer greater than or equal to 1."

        gradient_prompt = gradient_prompt_template.format(
            cur_prompt=cur_prompt, example_string=example_string
        )

        if nb_gradient_samples == 1:
            gradient, logging_dict = self.optim_model.generate(gradient_prompt)
        else:
            gradient_prompt_batch = [gradient_prompt] * nb_gradient_samples
            gradient_batch, logging_dict = self.optim_model.batch_forward_func(
                gradient_prompt_batch
            )
            gradient_summary_prompt = self._get_gradient_summary_prompt(gradient_batch)
            gradient, _ = self.optim_model.generate(gradient_summary_prompt)

            if self.print_log and self.logger is not None:
                log_str = gradient_summary_template.format(
                    prompt=gradient_summary_prompt, summary=gradient
                )

                self.logger.info(log_str)

        self.tracker.log({f"{key}_optim_model": value for key, value in logging_dict.items()})

        if self.print_log and self.logger is not None:
            log_str = gradient_log_template.format(
                gradient_prompt=gradient_prompt, gradient=gradient
            )

            self.logger.info(log_str)

        return gradient, gradient_prompt

    def _clean_optim_response(self, optim_response):
        pattern = r"<START>(.*?)<END>"
        matches = re.findall(pattern=pattern, string=optim_response, flags=re.DOTALL)
        return [m.strip() for m in matches]

    def optimize(
        self,
        cur_prompt: str,
        correct_string: str,
        error_string: str,
        gradient_positive: str,
        gradient_negative: str,
        trajectory_prompts,
        steps_per_gradient,
        optimize_prompt_template,
    ):
        assert (
            error_string is not None or correct_string is not None
        ), "[Optimization Error] either error_string or correct_string should be provided."

        optimize_prompt = optimize_prompt_template.format(
            cur_prompt=cur_prompt,
            correct_string=correct_string,
            error_string=error_string,
            gradient_positive=gradient_positive,
            gradient_negative=gradient_negative,
            trajectory_prompts=trajectory_prompts,
            steps_per_gradient=steps_per_gradient,
        )

        response, logging_dict = self.optim_model.generate(optimize_prompt)
        self.tracker.log({f"{key}_optim_model": value for key, value in logging_dict.items()})

        optimized_prompt = self._clean_optim_response(response)
        if self.print_log and self.logger is not None:
            log_str = optimize_log_template.format(
                optimize_prompt=optimize_prompt,
                response=response,
                optimized_prompt=optimized_prompt,
            )
            self.logger.info(log_str)

        return optimized_prompt

    def gradient_descent_step(self, cur_prompt: str, batch, helper_data, depth=None):

        (
            gradient_positive,
            gradient_negative,
            gradient_positive_prompt,
            gradient_negative_prompt,
        ) = ("", "", "", "")
        if self.logger is not None:
            self.logger.info(f"cur_prompt: {cur_prompt}")

        forward_output = self.forward(batch=batch, cur_prompt=cur_prompt)
        correct_np = np.array(forward_output["correct"])
        if len(correct_np.shape) == 1:
            # this means we are dealing with single metric evaluation for each sample in the batch
            error_string, correct_string = self._split_error_and_correct_examples(
                forward_output=forward_output
            )
        elif len(correct_np.shape) == 2:
            # this means we are dealing with multiple metric evaluation for each sample in the batch
            error_string, correct_string = (
                self._split_error_and_correct_examples_multi_metric(
                    forward_output=forward_output
                )
            )
        else:
            raise ValueError(f"The correct_np shape is not valid: {correct_np.shape}")

        trajectory_prompts = helper_data["trajectory_prompts"]
        trajectory_prompts = self._build_prompt_trajectory_str(trajectory_prompts)

        if forward_output["acc"] == 1.0:
            optimize_prompt_template = self.ascend_optimize_prompt_template
            gradient_positive, gradient_positive_prompt = self.cal_gradient(
                cur_prompt=cur_prompt,
                example_string=correct_string,
                gradient_prompt_template=self.ascend_gradient_prompt_template,
                nb_gradient_samples=self.gradient_sampling,
            )
        elif forward_output["acc"] == 0.0:
            optimize_prompt_template = self.optimize_prompt_template
            gradient_negative, gradient_negative_prompt = self.cal_gradient(
                cur_prompt=cur_prompt,
                example_string=error_string,
                gradient_prompt_template=self.gradient_prompt_template,
                nb_gradient_samples=self.gradient_sampling,
            )
        else:
            gradient_negative, gradient_negative_prompt = self.cal_gradient(
                cur_prompt=cur_prompt,
                example_string=error_string,
                gradient_prompt_template=self.gradient_prompt_template,
                nb_gradient_samples=self.gradient_sampling,
            )
            if depth is not None and depth < self.positive_reinforcement_depth:
                optimize_prompt_template = self.optimize_prompt_template
            else:
                optimize_prompt_template = self.mix_optimize_prompt_template
                gradient_positive, gradient_positive_prompt = self.cal_gradient(
                    cur_prompt=cur_prompt,
                    example_string=correct_string,
                    gradient_prompt_template=self.ascend_gradient_prompt_template,
                    nb_gradient_samples=self.gradient_sampling,
                )

        optimized_prompts = self.optimize(
            cur_prompt=cur_prompt,
            correct_string=correct_string,
            error_string=error_string,
            gradient_positive=gradient_positive,
            gradient_negative=gradient_negative,
            trajectory_prompts=trajectory_prompts,
            steps_per_gradient=self.num_new_prompts,  # number of new prompts to generate from the gradient
            optimize_prompt_template=optimize_prompt_template,
        )

        gradient_descent_output = forward_output
        gradient_descent_output["example_string"] = self._format_tagged_output(
            [("error examples", error_string), ("correct examples", correct_string)]
        )

        gradient_descent_output["gradient"] = self._format_tagged_output(
            [
                ("positive gradient", gradient_positive),
                ("negative gradient", gradient_negative),
            ]
        )
        gradient_descent_output["gradient_prompt"] = self._format_tagged_output(
            [
                ("positive gradient prompt", gradient_positive_prompt),
                ("negative gradient prompt", gradient_negative_prompt),
            ]
        )
        gradient_descent_output["optimized_prompts"] = optimized_prompts
        return gradient_descent_output

    def _format_tagged_output(self, tag_value_pairs):
        """
        Create a formatted string with values wrapped in XML-style tags.

        Args:
            tag_value_pairs: A list of (tag_name, value) tuples

        Returns:
            A formatted string with all values wrapped in their respective tags
        """

        return "\n".join(
            [f"<{tag}>\n{value}\n</{tag}>" for tag, value in tag_value_pairs]
        ).strip()

    def __call__(self, batch, cur_prompt: str, helper_data=None, depth=None):
        return self.gradient_descent_step(
            cur_prompt=cur_prompt, batch=batch, helper_data=helper_data, depth=depth
        )
