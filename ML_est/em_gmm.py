import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import os

class EM_GMM():

    def __init__(self, x, num_classes=4, max_iter=50, threshold=0.00001):
        
        self.num_classes = num_classes
        self.num_samples = x.shape[0]
        self.x_dim = x.shape[1]

        self.max_iter = max_iter
        self.threshold = threshold

        self.init_params()
        
    
    def init_params(self):

        
        self.mu = np.array(np.random.randn(self.num_classes, self.x_dim))
        self.sigma = np.array([np.identity(self.x_dim) for i in range(self.num_classes)])
        self.pi = np.array([1/self.num_classes for i in range(self.num_classes)])
        
    def calc_gaussian(self, x, mu, sigma):
        exp = -0.5 * np.diag((x - mu) @ np.linalg.inv(sigma) @ (x - mu).T)
        deno = (np.sqrt(2 * np.pi) ** self.x_dim) * np.sqrt(np.linalg.det(sigma))
        return np.exp(exp)/deno
    
    def calc_mix_gaussian(self, x):
        return np.array([self.pi[i] * self.calc_gaussian(x, self.mu[i], self.sigma[i]) \
                                                    for i in range(self.num_classes)])
    def calc_log_likelihood(self, x):
        mix_gaussian = self.calc_mix_gaussian(x)
        return np.sum(np.log(mix_gaussian.sum(axis=0)))

    def output_result(self, gamma):

        posterior = np.round(gamma.T, decimals=3)
        with open('z.csv', 'w+') as f:
            writer = csv.writer(f)
            writer.writerows(posterior)
        
        with open('params.dat', 'w+') as f:
            f.writelines("Mean Vector: \n")
            f.writelines(str(self.mu) + "\n")
            f.writelines("Covariance Matrix: \n")
            f.writelines(str(self.sigma) + "\n")
            f.writelines("Ratio: \n ")
            f.writelines(str(self.pi) + "\n")
    
        return

    def exec(self, x):

        last_ll = self.calc_log_likelihood(x)

        for i in range(self.max_iter):

            mix_gaussian = self.calc_mix_gaussian(x)
            gamma = mix_gaussian/mix_gaussian.sum(axis=0)
            self.pi = gamma.sum(axis=1)[:,None]/gamma.sum()
            self.mu = (gamma @ X) / gamma.sum(axis=1)[:,None]

            sigma_list = []

            for k in range(self.num_classes):
                sigma_k = np.zeros([self.x_dim, self.x_dim])
                for n in range(self.num_samples):
                    sigma_k += ((x[n]-self.mu[k])[:,None]@(x[n]-self.mu[k])[None,:]) * gamma[k][n]
                sigma_k = sigma_k/gamma.sum(axis=1)[k]
                sigma_list.append(sigma_k)
            
            self.sigma = np.array(sigma_list)

            now_ll = self.calc_log_likelihood(x)
            gap_ll = np.abs(now_ll - last_ll) 
    
            print("Iteration: " + str(i+1))
            print("Log Likelihood: " + str(now_ll))
            print("Gap: " + str(gap_ll))

            if gap_ll/self.num_samples < self.threshold:
                self.output_result(gamma)
                return 
            else:
                last_ll = now_ll

        print("EM algorthm achieved max iteration.")
        gamma = mix_gaussian/mix_gaussian.sum(axis=0)
        self.output_result(gamma)

        return
            
    
input_data = sys.argv[1]
output_data = sys.argv[2]
output_params = sys.argv[3]

with open(input_data) as f:
    reader = csv.reader(f)
    X = [row for row in reader]
    X = np.float_(X)

model = EM_GMM(X, 4)
model.exec(X)

# for  i in range(2, 10):
#     model = EM_GMM(X, i)
#     ll = model.exec(X)
#     print("Number of classes: " + str(i))
#     print("Log Likelihood: " + str(ll))
