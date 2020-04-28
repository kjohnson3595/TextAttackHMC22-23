from .attack_result import AttackResult
from textattack.shared import utils

class FailedAttackResult(AttackResult):
    def __init__(self, original_result, perturbed_result=None):
        perturbed_result = perturbed_result or original_result
        super().__init__(original_result, perturbed_result)

    def str_lines(self, color_method=None):
        lines = (self.goal_function_result_str(color_method), self.original_text.text)
        return tuple(map(str, lines))

    def goal_function_result_str(self, color_method=None):
        failed_str = utils.color_text('[FAILED]', 'red', color_method)
        return utils.color_text(self.original_output, method=color_method) + '-->' + failed_str 
