import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import os
from scipy.special import digamma, logsumexp

np.random.seed(200)

class VB_GMM():

    def __init__(self, x, num_classes=5, max_iter=100, threshold=0.00001):

        self.num_classes = num_classes
        self.num_samples = x.shape[0]
        self.x_dim = x.shape[1]

        self.max_iter = max_iter
        self.threshold = threshold

        self.init_params()
        
    
    def init_params(self):

        self.m0 = np.random.randn(self.x_dim)
        self.W0 = np.identity(self.x_dim)
        self.alpha0 = 0.01
        self.beta0 = 1.0
        self.nu0 = self.x_dim

        self.m = np.random.randn(self.num_classes, self.x_dim)
        self.W = np.tile(self.W0, (self.num_classes, 1, 1))
        self.alpha = np.ones(self.num_classes) * self.alpha0
        self.beta = np.ones(self.num_classes) * self.beta0
        self.nu = np.ones(self.num_classes) * self.nu0

        self.mu = self.m
        self.sigma = np.array([np.linalg.inv(self.nu[k] * self.W[k]) for k in range(self.num_classes)])
        self.pi = np.array([1/self.num_classes for i in range(self.num_classes)])
    
    def calc_gaussian(self, x, mu, sigma):
        exp = -0.5 * np.diag((x - mu) @ np.linalg.inv(sigma) @ (x - mu).T)
        deno = (np.sqrt(2 * np.pi) ** self.x_dim) * np.sqrt(np.linalg.det(sigma))
        return np.exp(exp)/deno
    
    def calc_mix_gaussian(self, x, mu, sigma, pi):
        return np.array([pi[i] * self.calc_gaussian(x, mu[i], sigma[i]) \
                                                    for i in range(self.num_classes)])
    def calc_log_likelihood(self, x):
        self.mu = self.m
        for i in range(self.num_classes):
            self.sigma[i] = np.linalg.inv(self.nu[i]*self.W[i])
        self.pi = self.alpha / np.sum(self.alpha, keepdims=True)
        mix_gaussian = self.calc_mix_gaussian(x, self.mu, self.sigma, self.pi)
        return np.sum(np.log(mix_gaussian.sum(axis=0)))
    
    def output_result(self, gamma):

        posterior = np.round(gamma, decimals=3)
        with open('z.csv', 'w+') as f:
            writer = csv.writer(f)
            writer.writerows(posterior)
        
        with open('params.dat', 'w+') as f:
            f.writelines("alpha: \n")
            f.writelines(str(self.alpha) + "\n")
            f.writelines("beta: \n")
            f.writelines(str(self.beta) + "\n")
            f.writelines("m: \n ")
            f.writelines(str(self.m) + "\n")
            f.writelines("nu: \n ")
            f.writelines(str(self.nu) + "\n")
            f.writelines("W: \n ")
            f.writelines(str(self.W) + "\n")
    
        return
    
    def e_step(self, x):

        log_pi = digamma(self.alpha) - digamma(self.alpha.sum())
        log_sigma = np.sum([digamma((self.nu + 1 - d)/2) for d in range(self.x_dim)], axis=0) \
                + self.x_dim * np.log(2) + np.log(np.linalg.det(self.W))

        dif = np.tile(x[:,None,None,:],(1,self.num_classes,1,1))-np.tile(self.m[None,:,None,:], (self.num_samples,1,1,1))
        dif = np.tile(self.nu[None,:], (self.num_samples,1)) \
                    * (dif @ np.tile(self.W[None,:,:,:],(self.num_samples,1,1,1)) @ dif.transpose(0,1,3,2))[:,:,0,0]
        
        log_rho = log_pi + 0.5 * log_sigma - 0.5 * ((self.x_dim / self.beta) + dif)
        log_gamma = log_rho - logsumexp(log_rho,axis=1)[:,None]
        gamma = np.exp(log_gamma)
        return gamma

    def m_step(self, x, gamma):

        S_k = gamma.sum(axis=0)
        S_k_x = gamma.T @ x
        S_k_xx = []

        for k in range(self.num_classes):
            x_mul= np.zeros([self.x_dim, self.x_dim])
            for n in range(self.num_samples):
                x_mul += x[n][:,None]@x[n][None,:] * gamma[n][k]
            S_k_xx.append(x_mul)
        S_k_xx = np.array(S_k_xx)
                
        self.alpha = self.alpha0 + S_k
        self.beta = self.beta0 + S_k
        self.nu = self.nu0 + S_k
        self.m = (self.beta0 * self.m0 + S_k_x) / self.beta[:,None]

        for k in range(self.num_classes):
            m0_mul = self.beta0 * (self.m0[:,None]@self.m0[None,:])
            m_mul = self.beta[k] * (self.m[k][:,None]@self.m[k][None,:])
            W_inv_k = np.linalg.inv(self.W0) + m0_mul + S_k_xx[k] - m_mul
            self.W[k] = np.linalg.inv(W_inv_k)
        
        return 
        
    def exec(self, x):

        last_ll = self.calc_log_likelihood(x)

        for i in range(self.max_iter):

            gamma = self.e_step(x)
            self.m_step(x, gamma)

            now_ll = self.calc_log_likelihood(x)
            gap_ll = np.abs(now_ll - last_ll) 
    
            print("Iteration: " + str(i+1))
            print("Log Likelihood: " + str(now_ll))
            print("Gap: " + str(gap_ll))

            if gap_ll/self.num_samples < self.threshold:
                gamma = self.e_step(x)
                self.output_result(gamma)
                return 
            else:
                last_ll = now_ll

        print("VB algorthm achieved max iteration.")
        gamma = self.e_step(x)
        self.output_result(gamma)
        return



input_data = sys.argv[1]
output_data = sys.argv[2]
output_params = sys.argv[3]

with open(input_data) as f:
    reader = csv.reader(f)
    X = [row for row in reader]
    X = np.float_(X)
    
model = VB_GMM(X, 4)
model.exec(X)