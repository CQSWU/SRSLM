class InferenceCorrection:
    @staticmethod
    def gate(entropy):
        return (entropy > 0.01).unsqueeze(-1)

    def apply(self, base_logits, residual, entropy):
        gate = self.gate(entropy).to(base_logits)
        learned_delta = gate * residual
        return base_logits + learned_delta
