class Config:

    __conf = {
        "random_seed": 42,
    }

    __setters = []

    @staticmethod
    def value(name):
        return Config.__conf[name]

    @staticmethod
    def set(name, value):
        if name in App.__setters:
            App.__conf[name] = value
        else:
            raise NameError("This config value is not settable.")
