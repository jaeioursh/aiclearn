import tensorflow as tf 
import numpy as np 
from collections import deque
from random import sample

class noise:
    def __init__(self,size,th=.3*2,sig=.15*2,mu=0.0,dt=1e-2):
        self.size=size
        
        self.th=th
        self.dt=dt 
        self.sig=sig 
        self.mu=mu  
        self.reset()
    
    def sample(self):
        
        #return 0.2*np.random.normal(size=self.size)
        self.state=self.state \
        + self.th*(self.mu-self.state)*self.dt \
        + self.sig*(self.dt**0.5)*np.random.normal(size=self.size)
        out=self.state.copy()
        #out[0]+=1 # forward vel
        out[1]*=0.5 # angled vel
        return out

    def reset(self):
        self.state=np.zeros(self.size)


class agent:
    def __init__(self, sess, s_dim, a_dim, hc_dim,ha_dim, lr, batch,gamma,dueling=0):
        self.sess=sess
        self.tau=0.001
        self.s_dim=s_dim
        self.a_dim=a_dim
        self.batch=batch
        self.gamma=gamma
        self.dueling=dueling

        self.base_line=False

        self.var=0.05
        self.a_hidden=ha_dim
        self.c_hidden=hc_dim

        self.s=tf.placeholder(tf.float32,[None,s_dim])
        self.a=tf.placeholder(tf.float32,[None,a_dim])
        self.r=tf.placeholder(tf.float32,[None,1])

        self.hist=[deque(maxlen=100000) for i in range(2)]

        self.actor0, self.a_params0  = self.gen_actor()
        self.critic0,self.c_params0 = self.gen_critic()


        self.actor1, self.a_params1  = self.gen_actor()
        self.critic1,self.c_params1 = self.gen_critic()


        self.tactor0, self.ta_params0  = self.gen_actor()
        self.tcritic0,self.tc_params0 = self.gen_critic()


        self.tactor1, self.ta_params1  = self.gen_actor()
        self.tcritic1,self.tc_params1 = self.gen_critic()

        self.actor, self.a_params = [self.actor0,self.actor1], [self.a_params0, self.a_params1]
        self.critic,self.c_params = [self.critic0,self.critic1],[self.c_params0,self.c_params1]

        self.tactor, self.ta_params = [self.tactor0,self.tactor1], [self.ta_params0, self.ta_params1]
        self.tcritic,self.tc_params = [self.tcritic0,self.tcritic1],[self.tc_params0,self.tc_params1]

        if self.dueling:
            self.action_grads = [tf.gradients(self.critic[i]-self.critic[j], self.a) for i,j in [[0,1],[1,0]]  ]
        else:
            self.action_grads = [tf.gradients(self.critic[i], self.a) for i in range(2)  ]


        self.a_grad= tf.placeholder(tf.float32, [None, self.a_dim])

        self.actor_gradients_ = [tf.gradients(self.actor[i], self.a_params[i], -self.a_grad) for i in range(2) ]
        self.actor_gradients = [list(map(lambda x: tf.div(x, self.batch), self.actor_gradients_[i]))  for i in range(2)]
        

        self.a_opt = [tf.train.AdamOptimizer(lr*0.1).apply_gradients(zip(self.actor_gradients[i], self.a_params[i]))  for i in range(2)]
        
        
        self.loss = [tf.losses.mean_squared_error(self.critic[i],self.r) for i in range(2)]
        self.c_opt = [tf.train.AdamOptimizer(lr).minimize(self.loss[i]) for i in range(2)]

        self.update_ta = \
            [[self.ta_params[j][i].assign(tf.multiply(self.a_params[j][i], self.tau) + \
            tf.multiply(self.ta_params[j][i], 1. - self.tau))
            for i in range(len(self.a_params[0]))] for j in range(2)]
        
        self.update_tc = \
            [[self.tc_params[j][i].assign(tf.multiply(self.c_params[j][i], self.tau) + \
            tf.multiply(self.tc_params[j][i], 1. - self.tau))
            for i in range(len(self.c_params[0]))] for j in range(2)]

    def activate(self,x):
        return tf.nn.tanh(x)

    def gen_critic(self):
        with tf.name_scope("critic") as scope:
            w1=tf.Variable(tf.random_normal([self.s_dim, self.c_hidden],stddev=self.var))
            b1=tf.Variable(tf.random_normal([self.c_hidden],stddev=self.var))
            netc=tf.matmul(self.s,w1)+b1

            w2=tf.Variable(tf.random_normal([self.a_dim, self.c_hidden],stddev=self.var))
            #b2=tf.Variable(tf.random_normal([self.c_hidden],stddev=self.var))
            neta=tf.matmul(self.a,w2)
            
            net=self.activate(neta+netc)



            w3=tf.Variable(tf.random_normal([self.c_hidden,1],stddev=self.var))
            b3=tf.Variable(tf.random_normal([1],stddev=self.var))
            #net=self.activate(tf.matmul(net,w3)+b3)
            net=tf.matmul(net,w3)+b3
            return net, [w1,w2,w3,b1,b3]


    def gen_actor(self):
        with tf.name_scope("actor") as scope:
            w1=tf.Variable(tf.random_normal([self.s_dim, self.a_hidden],stddev=self.var))
            b1=tf.Variable(tf.random_normal([self.a_hidden],stddev=self.var))
            #net=self.activate(tf.matmul(self.s,w1)+b1)

            net=self.activate(tf.matmul(self.s,w1)+b1)
            
            w2=tf.Variable(tf.random_normal([self.a_hidden,self.a_dim],stddev=self.var))
            b2=tf.Variable(tf.random_normal([self.a_dim],stddev=self.var))
            net=self.activate(tf.matmul(net,w2)+b2)

            return net, [w1,w2,b1,b2]

    def store(self,s,a,r,sp,done,idx):
        self.hist[idx].append([s,a,r,sp,done])
        #for h in zip(s,a,r,sp,done):
        #    self.hist[idx].append(h)

    def actor_train(self,s,a,idx):
        grads=self.sess.run(self.action_grads[idx],feed_dict={self.s:s,self.a:a})
        #print(grads)
        self.sess.run(self.a_opt[idx],feed_dict={self.s:s,self.a_grad:grads[0]})

    def critic_train(self,s,a,r,idx):
        _,loss = self.sess.run([self.c_opt[idx],self.loss[idx]],feed_dict={self.s:s,self.a:a,self.r:r})
        return loss

    def train_all(self,idx):
        if len(self.hist[idx])<self.batch:
            return 0.0,0.0,0.0
        hist=sample(self.hist[idx],self.batch)
        S,A,R,SP,DONE=[],[],[],[],[]
        for s,a,r,sp,d in hist:
            S.append(s)
            A.append(a)

            
            R.append(r)
            SP.append(sp)
            DONE.append(not d)

        AP=self.sess.run(self.tactor[idx],feed_dict={self.s:S})
        RP=self.sess.run(self.tcritic[idx],feed_dict={self.s:SP,self.a:AP})
        RP_=self.sess.run(self.critic[idx],feed_dict={self.s:SP,self.a:AP})
        DONE =np.array(DONE,dtype=bool)
        R=np.array(R)
        RP=RP.T[0]

        R[DONE]+=self.gamma*RP[DONE]

        R=np.array([R]).T
        
        L=self.critic_train(S,A,R,idx)
        self.actor_train(S,A,idx)

        self.sess.run(self.update_ta[idx])
        self.sess.run(self.update_tc[idx])
        #print(np.mean(RP))
        return L,np.mean(RP),np.mean(RP_)

    def act(self,s,idx):
        return self.sess.run(self.actor[idx],feed_dict={self.s:s})

    def save(self,fname):
        saver=tf.train.Saver()
        saver.save(self.sess,fname+".ckpt")

    def load(self,fname):
        saver=tf.train.Saver()
        saver.restore(self.sess,fname+".ckpt")

if __name__ =="__main__":
    import gym 
    import matplotlib.pyplot as plt
    env = gym.make('Pendulum-v0')
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    print(env.action_space.high)
    with tf.Session() as sess:
        
        
        bot=agent(sess,s_dim,a_dim,40,40,.001,32,.99,0)
        n=noise(a_dim)

        init=tf.global_variables_initializer()
        sess.run(init)
        env.seed(0)
        RR=[]
        for i in range(1000):
           
            
            s=env.reset()
                
            s=np.array([s])
            R=[]
            L=0.0
            arr,arr2=0.0,0.0
            for j in range(100):
                #print(j,s)
                #if i%100==0:
                #    env.render()
                a_noise=n.sample()
                #print(a_noise)
                a=bot.act(s,0)+a_noise
                sp, r, done, info = env.step(a[0]*2.0)
                R.append(r)
                sp=np.array([sp])
                bot.store(s[0],a[0],r,sp[0],done,0)
                l,r_,r_2=bot.train_all(0)
                L+=l
                arr+=r_
                arr2+=r_2
                s=sp
            print(i,sum(R),L,arr,arr2)
            RR.append(sum(R))
        plt.plot(RR)
        plt.show()