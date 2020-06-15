# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:05:25 2020

@author: Frederik
"""

import numpy as np
from scipy.linalg import svd
import random

class sphere_statistics:
    def __init__(self, **kwargs):
        
        data_type = kwargs.get("data_type")
        
        if data_type == "Own":
            self.data = self.sphere_check(kwargs.get("data"), kwargs.get("n"))
            self.data_num = kwargs.get("n")
        elif data_type == "Geodesic_curve":
            self.data = self.sim_geodesic(kwargs.get("n"),kwargs.get("p"),
                                            kwargs.get("q"))
            self.data_num = kwargs.get("n")
        elif data_type == "Geodesic_ball":
            self.data = self.sim_geodesic_balll(kwargs.get("n"),kwargs.get("p"),
                                            kwargs.get("r"))
            self.data_num = kwargs.get("n")
        elif data_type == "Random":
            self.data = self.sim_unif_sphere(kwargs.get("n"))
            self.data_num = kwargs.get("n")
        else:
            raise ValueError("Invalid input")
    
    def get_data(self):
        
        return self.data
    
    def change_data(self, data, n):
        
        dat = self.sphere_check(data, n)
        
        self.data = dat
        self.n = n
        
        return
    
    def sphere_check(self, data, n, eps=0.01):

        norm_data = data
        
        for j in range(0, n):
            norm_data[:,j] = data[:,j]/np.linalg.norm(data[:,j])
        
        return norm_data
    
    def Exp_map(self, p, v, t):
        
        #Exp_p(v) = p*cos(||v||)+v*sin(||v||)/||v||
        
        e_norm_v = np.linalg.norm(v)
        if e_norm_v*t > np.pi:
            print("Error when computing the exponential map. The length of the"+
                  " vector is longer than Pi!")
            return None
        
        Exp_map = p*np.cos(e_norm_v*t)+v*np.sin(e_norm_v*t)/e_norm_v
        
        return Exp_map
        
    def Log_map(self, p, x, eps = 0.01):
        
        #Log_p(x) = (x-p*((x^T)*p))*theta/sin(theta)
        #theta = arccos((x^T)*p)
        
        xT = np.transpose(x)
        xT_p = np.dot(xT, p)
        
        if (xT_p>1 and xT_p-eps<=1):
            theta = 0
        elif (xT_p<-1 and xT_p+eps>=-1):
            theta = np.pi
        else:
            theta = np.arccos(xT_p)
            
        if abs(theta) < eps:
            Log_map = x-np.dot(p, xT_p)
        else:
            Log_map = (x-np.dot(p, xT_p))*theta/np.sin(theta)
        
        return Log_map
    
    def rot_from_north(self, p, v, eps = 0.001):
        
        theta = np.arccos(p[2]/np.linalg.norm(p))
        
        if abs(p[1]) < eps:
            phi = np.pi/2
        else:
            phi = np.arctan(p[1]/p[0])
        
        RinvZ = np.array([[np.cos(-phi), np.sin(-phi),0],
                          [-np.sin(-phi), np.cos(-phi),0],
                          [0,0,1]])
        
        RinvY = np.array([[np.cos(theta),0,np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
        
        return RinvZ @ RinvY @ v
    
    def intrinsic_mean_gradient_search(self, tau, eps = 0.01, max_ite = 100, 
                                       initial_guess = np.array([0,0,0])):
        
        mu = np.zeros((3,max_ite+1))
        if (not initial_guess.any()):
            mu[:,0] = self.data[:,0]
        else:
            mu[:,0] = initial_guess
        
        j = 0
        while True:
            Log_sum = 0
            
            for i in range(0,self.data_num):
                Log_sum += self.Log_map(mu[:,j], self.data[:,i])
            
            delta_mu = tau/self.data_num*Log_sum
            mu[:,j+1] = self.Exp_map(mu[:,j], delta_mu, 1)
            
            if np.linalg.norm(mu[:,j+1]-mu[:,j])<eps:
                print("The Karcher Mean has succesfully been computed after j=" + str(j+1) +
                      " iterations with a tolerance of eps=" + str(eps))
                break
            elif j+1>max_ite:
                print("The algorithm has been stopped due to the maximal number of " +
                      "iterations of " + str(max_ite) + "!")
                break
            else:
                j += 1
                Log_sum = 0
                
        return mu[:,0:(j+1)]
    
    def pca(self, mu):
        
        X = np.zeros([self.data_num, 3])
        
        for j in range(0, self.data_num):
            X[j,:] = self.data[:,j]
            
        Y = X - np.ones((self.data_num,1))*X.mean(axis=0)
         
        U,S,V = svd(Y,full_matrices=False)
         
        return {'sigma': S, 'v_k': np.transpose(V), 'mean': X.mean(axis=0)}
                
    def pga(self, mu):
        
        u = np.zeros((3,self.data_num))
    
        X = np.zeros([self.data_num, 3])
                
        for j in range(0, self.data_num):
            u[:,j] = self.Log_map(mu, self.data[:,j])
            X[j,:] = u[:,j]
        
        Y = X - np.ones((self.data_num,1))*X.mean(axis=0)
        
        U,S,V = svd(Y,full_matrices=False)
        
        v_k = np.transpose(V)
        
        pga_component = np.zeros([3,3])
        
        for i in range(0,3):
            pga_component[:,i] = self.Exp_map(mu, v_k[:,i], 1)
        
        return {'lambda': S, 'v_k_log': v_k, 'v_k_exp': pga_component}
        
    def sim_unif_sphere(self, n):
        
        theta = np.random.uniform(-np.pi/2,np.pi/2, n)
        phi = np.random.uniform(-np.pi,np.pi, n)
        
        data_sim = np.array([np.sin(theta)*np.cos(phi), 
                             np.sin(theta)*np.sin(phi), np.cos(theta)])
        
        return data_sim
        
    def sim_geodesic_balll(self, n, p, r):
        
        data_sim = np.zeros((3,n))
        n_pole = np.array([0,0,1])
        
        theta = np.linspace(-np.pi,np.pi, n, endpoint = False)
        
        v = np.array([r*np.cos(theta), r*np.sin(theta), np.zeros(n)])
        
        for i in range(0,n):
            data_sim[:,i] = self.Exp_map(n_pole, v[:,i], 1)
            data_sim[:,i] = self.rot_from_north(p, data_sim[:,i])
        
        return data_sim
    
    def sim_geodesic(self, n, p, q):
        
        data_sim = np.zeros((3,n))
        time = np.linspace(0,1,n)
        
        v = self.Log_map(p, q)
        
        for j,t in enumerate(time, start = 0):
            data_sim[:,j] = self.Exp_map(p, v, t)
        
        return data_sim
    
    def geodesic_line(self, t, p, q):
        
        v = self.Log_map(p, q)
        
        e_norm_v = np.linalg.norm(v)
        
        v = np.pi*v/e_norm_v
        e_norm_v = np.pi
        
        Exp_map = np.zeros([3,len(t)])
        
        for i in range(0, len(t)):
            Exp_map[:,i] = p*np.cos(e_norm_v*t[i])+v*np.sin(e_norm_v*t[i])/e_norm_v
        
        return Exp_map
    
    def geodesic_line2(self, p, q, num = 100):
        
        v = self.Log_map(p, q)
        
        t = np.linspace(0,1,100)
        
        e_norm_v = np.linalg.norm(v)

        Exp_map = np.zeros([3,len(t)])
        
        for i in range(0, num):
            Exp_map[:,i] = p*np.cos(e_norm_v*t[i])+v*np.sin(e_norm_v*t[i])/e_norm_v
        
        return Exp_map
    
    def general_distance_k_means(self, K, clusters = [], MAX_ITE = 100, eps = 0.01, seed = None):
        
        random.seed(a=seed, version=2)
        
        index_list = range(0,self.data_num)
        split = int(self.data_num/K)
        loss_function = []
        gamma_rhs = []
        new_clusters = [[] for i in range(0,K)]
        data_clusters = [[] for i in range(0,K)]
        
        if not clusters:
            clusters = [[] for i in range(0,K)]
            
            for k in range(0,K):
                if k == K-1:
                    clusters[k] = index_list
                    #clusters[k] = index_list[k*split:]
                else:
                    choice = random.sample(index_list, split)
                    index_list = [index for index in index_list if index not in choice]
                    clusters[k] = choice
                    #clusters[k] = index_list[(k*int(split)):((k+1)*int(split))]
        
        loss_function.append(self.__k_means_loss_function(clusters, K))
                    
        j = 0
        while True:
            print(j)
            print(loss_function[j])
            for i in range(0, self.data_num):
                gamma_rhs = []
                for k in range(0,K):
                    gamma_rhs.append(self.__k_means_cluster_function(clusters[k], 
                                                                     self.data[:,i]))
                                     
                new_clusters[np.argmin(gamma_rhs)].append(i)
                
            clusters = new_clusters
            loss_function.append(self.__k_means_loss_function(clusters, K))
            
            j += 1
            if np.abs(loss_function[j]-loss_function[j-1])<eps:
                print("Clustering has ended due to convergence with epsilon =", eps)
                break
            elif j>=MAX_ITE:
                print("Clustering has ended due to the maximum number of iteration set to", 
                      MAX_ITE)
                break     
            else:
                new_clusters = [[] for i in range(0,K)]
        
        for k in range(0,K):
            data_clusters[k] = self.data[:,clusters[k]]
            
        val = {'Loss function' : loss_function, 'Index Clusters' : clusters,
               'Data Clusters' : data_clusters}
        
        return val
    
    def distance(self, x, y, eps = 0.01):
        
        xT = np.transpose(x)
        xT_p = np.dot(xT, y)
        
        if (xT_p>1 and xT_p-eps<=1):
            theta = 0
        elif (xT_p<-1 and xT_p+eps>=-1):
            theta = np.pi
        else:
            theta = np.arccos(xT_p)
        
        return theta
    
    def __k_means_cluster_function(self, cluster, x):
        
        n = len(cluster)
        sum1 = 0
        sum2 = 0
        
        if n == 0:
            return 0
        
        for j in range(0,n):
            sum1 += self.distance(x,self.data[:,cluster[j]])**2
            
        for j in range(0,n):
            for i in range(0,n):
                sum2 += self.distance(self.data[:,cluster[j]],self.data[:,cluster[i]])**2
                
        return 2*sum1/n-sum2/(n**2)
    
    def __k_means_loss_function(self, clusters, K):
        
        val = 0
        n = 0
        
        for k in range(0,K):
            cluster = clusters[k]
            n = len(cluster)
            for j in range(0,n):
                for i in range(0,n):
                    val += self.distance(self.data[:,cluster[j]],self.data[:,cluster[i]])**2
                    
        return val
        
    def rotate_data_to_north_pole(self, p, data, n, eps = 0.01):
        
        theta = np.arccos(p[2]/np.linalg.norm(p))
        
        if theta<eps:
            return data
        
        if abs(p[0]) < eps:
            phi = np.pi/2
        else:
            phi = np.arctan(p[1]/p[0])

        
        Rz = np.array([[np.cos(phi), np.sin(phi), 0],
              [-np.sin(phi), np.cos(phi), 0],
              [0,0,1]])
        
        if (Rz @ p)[0]<0:
            Ry = np.array([[np.cos(theta),0, np.sin(theta)],
                           [0, 1, 0],
                           [-np.sin(theta), 0, np.cos(theta)]])
        else:
            Ry = np.array([[np.cos(theta),0, -np.sin(theta)],
                           [0, 1, 0],
                           [np.sin(theta), 0, np.cos(theta)]])
        
        rot_data = np.zeros([3,n])
        
        for i in range(0,n):
            rot_data[:,i] = Ry @ Rz @ data[:,i]
        
        return rot_data
    
    def __initial_centroids(self, K, *args):
        
        if args:
            random.seed(a=args[0], version=2)
        
        index_list = range(0,self.data_num)
        centroids = self.data[:,random.sample(index_list, K)]
        
        return centroids
    
    def distance_matrix(self, data1, data2, dist_fun):
        
        K = data1.shape[1]
        N = data2.shape[1]
        
        dist_matrix = np.zeros([K,N])
        
        for k in range(0,K):
            for n in range(0,N):
                dist_matrix[k,n] = dist_fun(data1[:,k], data2[:,n])**2
        
        return dist_matrix
    
    def __k_means_loss_function_classic(self, dist_mat, clusters_index_matrix):
        
        return (dist_mat @ clusters_index_matrix).diagonal().sum()
        
    def k_means_riemannian_manifold(self, K, eps = 0.001, rep = 10, **kwargs):
        
        seed_val = kwargs['seed']
        return_val = [None]*rep
        return_loss = [None]*rep
        
        for r in range(0,rep):
            centroids = np.zeros([3,K])
            data_clusters = [[] for i in range(0,K)]
            loss_function = []
            
            i = 0
            loss_function.append(0)
            centroids = self.__initial_centroids(K, seed_val+r)
            index_matrix = np.zeros([self.data_num, K])
            dist_mat = self.distance_matrix(centroids, self.data, self.distance)
            
            while True:            
                arg_dist = np.argmin(dist_mat, axis = 0)
                for n in range(0,self.data_num):
                    index_matrix[n, arg_dist[n]] = 1
                    
                for k in range(0,K):
                    data_clusters[k] = self.data[:,index_matrix[:,k]==1]
                    centroids[:,k] = self.__cluster_karcher_mean(data_clusters[k],
                                                               data_clusters[k].shape[1])
                    
                dist_mat = self.distance_matrix(centroids, self.data, self.distance)
                    
                loss_function.append(self.__k_means_loss_function_classic(dist_mat, index_matrix))
                
                if np.abs(loss_function[i+1]-loss_function[i])<eps:
                    break
                else:
                    index_matrix = np.zeros([self.data_num, K])
                    i += 1
            
            return_loss[r] = loss_function[-1]
            return_val[r] = {'Loss function' : loss_function, 'Index Clusters' : index_matrix,
               'Data Clusters' : data_clusters, 'Centroids': centroids}
        
        return return_val[np.argmin(return_loss)]
    
    def k_means_euclidean(self, K, eps = 0.001, rep = 10, **kwargs):
        
        seed_val = kwargs['seed']
        return_val = [None]*rep
        return_loss = [None]*rep
        
        for r in range(0,rep):
            centroids = np.zeros([3,K])
            data_clusters = [[] for i in range(0,K)]
            loss_function = []
            
            i = 0
            loss_function.append(0)
            centroids = self.__initial_centroids(K, seed_val+r)
            index_matrix = np.zeros([self.data_num, K])
            dist_mat = self.distance_matrix(centroids, self.data, self.__euclidean_distance)
            
            while True:            
                arg_dist = np.argmin(dist_mat, axis = 0)
                for n in range(0,self.data_num):
                    index_matrix[n, arg_dist[n]] = 1
                    
                for k in range(0,K):
                    data_clusters[k] = self.data[:,index_matrix[:,k]==1]
                    centroids[:,k] = np.mean(data_clusters[k], axis = 1)
                    
                dist_mat = self.distance_matrix(centroids, self.data, self.__euclidean_distance)
                    
                loss_function.append(self.__k_means_loss_function_classic(dist_mat, index_matrix))
                
                if np.abs(loss_function[i+1]-loss_function[i])<eps:
                    break
                else:
                    index_matrix = np.zeros([self.data_num, K])
                    i += 1
            
            return_loss[r] = loss_function[-1]
            return_val[r] = {'Loss function' : loss_function, 'Index Clusters' : index_matrix,
               'Data Clusters' : data_clusters, 'Centroids': centroids}
        
        return return_val[np.argmin(return_loss)]
        
    def __euclidean_distance(self, x, y):
        
        return np.linalg.norm(x-y)
        
    def __cluster_karcher_mean(self, data, n, tau = 1, eps = 0.001, max_ite = 100, 
                                       initial_guess = np.array([0,0,0])):
        
        mu = np.zeros((3,max_ite+1))
        if (not initial_guess.any()):
            mu[:,0] = data[:,0]
        else:
            mu[:,0] = initial_guess
        
        j = 0
        while True:
            Log_sum = 0
            
            for i in range(0,n):
                Log_sum += self.Log_map(mu[:,j], data[:,i])
            
            delta_mu = tau/self.data_num*Log_sum
            mu[:,j+1] = self.Exp_map(mu[:,j], delta_mu, 1)
            
            if np.linalg.norm(mu[:,j+1]-mu[:,j])<eps:
                break
            elif j+1>max_ite:
                break
            else:
                j += 1
                Log_sum = 0
        
        return mu[:,j+1]
        
    def tangent_space(self, p, *args):
        
        num = len(args)
        log_args = [[] for j in range(0,num)]
        
        for j in range(0,num):
            data = args[j]
            try:
                n = data.shape[1]
            except:
                n = 1   
                data.shape = (3,1)                                 
            log_data = np.zeros([3,n])
            for i in range(0,n):
                log_data[:,i] = self.Log_map(p, data[:,i])
                
            log_args[j] = log_data

        for j in range(0,num):
            log_args[j] = self.rotate_data_to_north_pole(p, log_args[j], log_args[j].shape[1])
            log_args[j] = np.delete(log_args[j], 2, 0)
        
        return log_args
        
    def haussdorf_distance(self, cluster1, cluster2):
        
        n = len(cluster1)
        m = len(cluster2)
        cluster_dist = np.zeros([n,m])
        
        for i in range(0,n):
            for j in range(0,m):
                dist_matrix = self.distance_matrix(cluster1[i], cluster2[j], self.distance)
                cluster_dist[i,j] = max([np.max(np.min(dist_matrix, axis = 0)),
                                     np.max(np.min(dist_matrix, axis = 1))])
        
        return cluster_dist
            
            
            
            
            
            
        
        
        
        
        
        
        
            
        
        
        