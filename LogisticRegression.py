import numpy as np
class LogisticRegression:

    def __init__(self, learning_rate=0.1, n_epochs=200, random_state=3):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.theta = None
        self.rng = np.random.default_rng(random_state)
        self.loss_list = None
        self.theta_list = None

    @staticmethod
    def sigmoid(z):
        value = 1/ (1+np.exp(-z))
        return value
    
    @staticmethod
    def ce_loss(data_X:np.ndarray, data_Y:np.ndarray, theta:np.ndarray):
        sig_input = np.matmul(data_X, theta)
        pred_y = LogisticRegression.sigmoid(np.matmul(data_X, theta))
        loss_term1 = data_Y*np.log2(pred_y)
        loss_term1 = np.nan_to_num(loss_term1, posinf=100, neginf=-100)
        loss_term2 = (1-data_Y)*np.log2(1-pred_y)
        loss_term2 = np.nan_to_num(loss_term2, posinf=100, neginf=-100)
        output = -np.mean(loss_term1 + loss_term2)
        return output


    @staticmethod
    def grad_loss(data_X:np.ndarray, data_Y:np.ndarray, theta:np.ndarray):
        pred_true_diff = LogisticRegression.sigmoid(np.matmul(data_X, theta)) - data_Y
        gradient = (1/data_X.shape[0]) * np.matmul(data_X.transpose(), pred_true_diff)
        return gradient



    def fit(self, data_X:np.ndarray, data_Y:np.ndarray):
        self.theta = self.rng.random((data_X.shape[1], 1))
        data_y_reshape = data_Y.reshape((data_Y.shape[0], 1))
        loss_list = [LogisticRegression.ce_loss(data_X, data_y_reshape, self.theta)]
        theta_list = [self.theta]
        for i in range(self.n_epochs):
            gradient = LogisticRegression.grad_loss(data_X, data_y_reshape, self.theta)
            theta_update = self.theta - self.learning_rate * gradient
            loss_list.append(LogisticRegression.ce_loss(data_X, data_y_reshape, theta_update))
            theta_list.append(theta_update)
            self.theta = theta_update
            if i%150 == 0:
                print(f'epoch{i}')
        self.loss_list = loss_list
        self.theta_list = theta_list
    
    def predict_proba(self, data_X:np.ndarray):
        sig_input = np.matmul(data_X, self.theta)
        return LogisticRegression.sigmoid(sig_input)

    def predict(self, data_X:np.ndarray, threshold=0.5):
        probabilities = self.predict_proba(data_X)
        predictions = np.zeros(probabilities.shape)
        predictions[probabilities >= threshold] = 1
        return predictions
