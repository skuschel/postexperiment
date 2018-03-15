
class Shot(dict):
    diagnostics = dict()

    def __getattr__(self, key):
        def call(*args, context=None, **kwargs):
            if context is None:
                context = dict()
            context['shot'] = self
            return self.diagnostics[key](self, *args, context=context, **kwargs)
        return call


