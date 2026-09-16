"""Strict source loading with one explicit additive architecture migration."""


class FinetuneWeights:
    @staticmethod
    def load(model, state, *, initialize_reference=False):
        if not initialize_reference:
            model.load_state_dict(state, strict=True)
            return
        expected = model.state_dict()
        fresh = {key: value for key, value in expected.items() if key.startswith('reference_head.')}
        missing = set(expected) - set(state)
        unexpected = set(state) - set(expected)
        if not fresh or missing != set(fresh) or unexpected:
            raise ValueError('Source checkpoint must match exactly apart from the complete new reference head; '
                             f'missing={sorted(missing)}, unexpected={sorted(unexpected)}')
        model.load_state_dict({**state, **fresh}, strict=True)
