import numpy as np

class ModelTemplate:
    """
    base model, other models can use as template:
    """

    def __init__(self, **kwargs):
        """
        kwargs: can be used for when models require additional parameters. 
        """
        raise NotImplementedError
    
    def predict(self, X_data: np.ndarray) -> list[str]:
        """
        X_data: Input data used to make prediction. Each row is one person's responses. Making
                this a NxD shaped matrix. N being the number of people who responded, D being the
                number of features we have.
        
        returns: A list of predictions for each row input, the first value is the prediction for the
                 first row, the second value is the prediction of the second row, e.t.c.
        
        """
        raise NotImplementedError