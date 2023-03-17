class assembly_regression():
    def __init__(self, regression_model, forescasting_model):
        self.regression_model = regression_model
        self.forescasting_model = forescasting_model
    def forwards(sellf, x):
        pass
    def predict(self, x):
        values = self.regression_model.predict(x)
        direction = self.forescasting_model.predict(x)
        values = values * direction
        return values