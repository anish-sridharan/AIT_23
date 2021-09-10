
#Q1
class Bayes:
    def __init__(self,hypothesis_list,prior_list, observations_list, likelihood_array):
        self.hypothesis_list_ = hypothesis_list
        self.prior_list_ = prior_list
        self.likelihood_array_ = likelihood_array
        self.observations_list_ = observations_list
        
    #Q2
    def likelihood(self,observation,hypothesis):
        
        pos_hyp = self.hypothesis_list_.index(hypothesis)
        #prior = self.prior_list_[pos_hyp]
        pos_obs = self.observations_list_.index(observation)
        lk= self.likelihood_array_[pos_hyp][pos_obs]
        return lk
    
    #Q3
    def norm_constant(self,observation):
        norm_cst = 0
        for item in self.hypothesis_list_:
            prior = self.prior_list_[self.hypothesis_list_.index(item)]
            norm_cst = norm_cst + prior*self.likelihood(observation,item)
        return norm_cst    

    #Q4
    def single_posterior_update(self,observation,priors_list):
        posterior_list = []
        count=0
        for prior in priors_list:

            likelihood = self.likelihood(observation,self.hypothesis_list_[count])
            const = self.norm_constant(observation)
            posterior = (prior*likelihood)/const
            posterior = round(posterior,3)
            posterior_list.append(posterior)
            count+=1
        return posterior_list    
    
    
    #Q5
    def compute_posterior(self,observations_list):
        posterior_list = []
        for observation in observations_list:
            posterior=self.single_posterior_update(observation,self.prior_list_)
            self.prior_list_=posterior
            
        return posterior

        







