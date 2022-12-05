import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# class for model controller with SGD

class Model_control_SGD(): 
    
    
    def __init__(self,raw_measurementdata):
        
       

         #initialize parameters
        self.initial_w = [ 0.00342657, -0.00053593,  0.00586184, -0.00018193]
        #[ 0,0,2.76566664,0.33001263]          #initial parameters for polynomial model
        
        self.learning_rate = 0.000034                              # learning rate for SGD update
        
        self.iteration = 1680                                   # number of iteration for running algorithm
        
        self.batchsize = 0                 
        
        self.total_shift = 82                                  # time delay 
        
        self.predicted_distance = 0                            #initial predicted distance
        
        self.counter = 0 
        
        self.system_SNR = 5                                   #system SNR for hydraulic piston system (dB)
        
        self.step_startcontrol = 320                           #step to start the control
        
        self.op_amplitude  = 3                                 # vibration amplitude 
        
        self.op_freq = 10                                      # vibration frequency
        
        self.sampling_freq = 5000                              #sampling frequency
        
        self.op_back_vel = 0                                   #background velocity
        
        self.interval_smooth_reg = 12
        
        self.interva_p = 40                                    #points interval for velocity calculation
        
        self.start_step_data = 4070                            #the time step to extract subdata from whole dataset
         
        self.velocity_data  = -10*(32768 - np.array(raw_measurementdata['Stellwert_Ventil'][self.start_step_data+self.total_shift:self.start_step_data+self.iteration+self.total_shift]))/32768  # get control data 
        print(self.velocity_data)
        self.control_data = 1*np.array(raw_measurementdata['vel_local'][self.start_step_data:self.start_step_data+self.iteration])                                      # get velocity data 
        
        self.measured_dist = np.array(raw_measurementdata['IST_weg_mm'][self.start_step_data:self.start_step_data+self.iteration])     # smoothed distance 

        
        self.smoothed_dist = np.array(raw_measurementdata['smoothed_dist'][self.start_step_data:self.start_step_data+self.iteration])     # smoothed distance 
#newton method to obtain numerical solution of model control output


    def newton_control_solver(self,starting_point,desired_vel,parameter_vector):
    
    #starting point,
        control = starting_point
        iteration = 0
        func_val = parameter_vector[0]*control**3 + parameter_vector[1]*control**2 + parameter_vector[2]*control + parameter_vector[3]
 
        numerical_error = abs(func_val-desired_vel)
        for j in range(1000): 
    #function value
            func_val = parameter_vector[0]*control**3 + parameter_vector[1]*control**2 + parameter_vector[2]*control - desired_vel + parameter_vector[3]
     #gradient val
            gradient_val = 3*parameter_vector[0]*control**2 + 2*parameter_vector[1]*control + 1*parameter_vector[2]
            control = control - (func_val /gradient_val)
        #solution error
            numerical_error = abs(func_val)
        
            iteration += 1
     #   print(iteration,numerical_error)
            if (numerical_error <= 0.001):
           
                newcontrol = control
          
                return newcontrol

# use stochastic gradient descent to train model
 #function to calculate predicted distance at certain time step
    def predicted_controlleroutput(self,dist_MTSdata,total_shift, local_timestep,predicted_velarray):
   
    
       predicted_distance = dist_MTSdata[local_timestep-total_shift] + np.sum(predicted_velarray)
    
       return predicted_distance

#desired distance

    def desired_dist(self,counter,smoothed_dist,total_shift, step_startcontrol, op_amplitude,op_freq, sampling_freq, op_back_vel):
   
    
        desired_distance = smoothed_dist[step_startcontrol] + op_amplitude*np.sin(2*np.pi*(op_freq*counter/sampling_freq) ) + op_back_vel* counter
        return desired_distance

# SGD optimizer for linear model

    def LinearSGD_optimizer(self,learning_rate,batchsize,local_vel,parameter_vector,current_control):

        w_1 =  parameter_vector[0]
        w_2 =  parameter_vector[1]
        w_3 =  parameter_vector[2]
        w_4 =  parameter_vector[3]
    
       #calculate gradient of error function

        gradient_subterm_w3 = local_vel*(w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control)
        gradient_subterm_w4 = w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control
  
       #update parameter via SGD
        parameter_vector[2] = parameter_vector[2] - 2*learning_rate*gradient_subterm_w3
        parameter_vector[3] = parameter_vector[3] - 2*learning_rate*gradient_subterm_w4 

    
#calculate control output from model
#design matrix
#    design_matrix = np.array([local_vel**3,local_vel**2,local_vel,1]).flatten()

        model_regout = parameter_vector[0]*local_vel**3 + parameter_vector[1]*local_vel**2 + parameter_vector[2]*local_vel + parameter_vector[3]
    
   # print(model_regout,new)
 #calculate error 
        error = abs((model_regout - current_control))
    
        return parameter_vector,error,model_regout


# SGD optimer for polynomial model

    def SGD_optimizer(self,learning_rate,batchsize,local_vel,parameter_vector,current_control):

        w_1 =  parameter_vector[0]
        w_2 =  parameter_vector[1]
        w_3 =  parameter_vector[2]
        w_4 =  parameter_vector[3]
    
        #calculate gradient of error function

        gradient_subterm_w1 = local_vel**3*(w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control)

        gradient_subterm_w2 = local_vel**2*(w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control)
        gradient_subterm_w3 = local_vel*(w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control)
        gradient_subterm_w4 = w_1*local_vel**3 + w_2*local_vel**2 + w_3*local_vel +w_4 - current_control
  
        #update parameter via SGD
        parameter_vector[0] = parameter_vector[0] - 2*learning_rate*gradient_subterm_w1
        parameter_vector[1] = parameter_vector[1] - 2*learning_rate*gradient_subterm_w2
        parameter_vector[2] = parameter_vector[2] - 2*learning_rate*gradient_subterm_w3
        parameter_vector[3] = parameter_vector[3] - 2*learning_rate*gradient_subterm_w4 

    
#calculate control output from model
#design matrix
#    design_matrix = np.array([local_vel**3,local_vel**2,local_vel,1]).flatten()

        model_regout = parameter_vector[0]*local_vel**3 + parameter_vector[1]*local_vel**2 + parameter_vector[2]*local_vel + parameter_vector[3]
    
   # print(model_regout,new)
 #calculate error 
        error = abs((model_regout - current_control))
    
        return parameter_vector,error,model_regout

    
    #calculate linear regression coefficient
    def estimate_coef(self,x, y):
    
    # number of observations/points
        n = np.size(x)
   
    # mean of x and y vector
        m_x = np.mean(x)
        m_y = np.mean(y)
 
    # calculating cross-deviation and deviation about x
        SS_xy = np.sum(y*x) - n*m_y*m_x
        SS_xx = np.sum(x*x) - n*m_x*m_x
 
    # calculating regression coefficients
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1*m_x

        return (b_1, b_0)
    #function to calculate movement of piston
    def movement_piston_simulation(self,interva_p,polynomial_model_regout_array,polynomial_model_control,parameter_vector,system_SNR):
        
        
        

        cleanmovement_simulation =  (parameter_vector[0]*polynomial_model_control**3 + parameter_vector[1]*polynomial_model_control**2 +parameter_vector[2]*polynomial_model_control + parameter_vector[3])
     
    #calculate noise variance for AWGN 
        noise_var =  np.var(np.array(polynomial_model_regout_array).flatten())/(10**((system_SNR / 10)))
    #calculate noisy simulated movement because of generated controller output from model control
        noisy_movement = np.random.normal(0, np.sqrt(noise_var), 1) + cleanmovement_simulation
        
        
        
        return noisy_movement
    def data_fitting_model_main(self,interval_smooth_reg,total_shift,measured_dist,system_SNR,interva_p,counter,step_startcontrol,op_amplitude,op_freq, sampling_freq, op_back_vel, predicted_distance,batchsize,initial_w,iteration,learning_rate,control_data,velocity_data,smoothed_dist):
    
    
        w1_array = []
        w2_array = []
        w3_array = []
        w4_array = []
    
        predicted_distancearray = []
        desired_distance_array = []
        error_array = []
        model_regout_array = []
        new_model_controlarray = []
        current_control_array = []
        vel_array = []
        polynomial_model_regout_array = []
        
        real_distancearray = []
        
        measured_distarray = []
        
        linearsmoothed_dist = smoothed_dist
        
        for j in range(iteration-batchsize):
            if (j == 0):
                parameter_vector = initial_w
                linearparameter_vector = initial_w 
                
            # at training stage
            if (j <= step_startcontrol):
                current_control = control_data[j]
                local_vel = velocity_data[j]
                real_distancearray.append(smoothed_dist[j])
            # at control stage
            else:
                #calculate current velocity
                current_control = (position_update - smoothed_dist[j-(interva_p)])/interva_p
               
                
              #rem assign control output with generated control output from polynomial model
                local_vel = polynomial_model_control
                
                real_distancearray.append(position_update )
                 
            
        
        
        # calculate desired dist
            if (j >= step_startcontrol):
                desired_distance = self.desired_dist(counter,smoothed_dist,total_shift, step_startcontrol, op_amplitude,op_freq, sampling_freq, op_back_vel)
                counter = counter + 1 
            else:
                desired_distance = smoothed_dist[j]
            
        #calculate predicted distance   
            newtotal_shift = int(0.5*total_shift)-6
            if (j > total_shift):
                predicted_velarray = np.array(model_regout_array[j-newtotal_shift:j]).flatten()
                predicted_distance = self.predicted_controlleroutput(smoothed_dist,newtotal_shift, j,predicted_velarray)
            #calculate velocity deviation
      
                vel_devi = (predicted_distance- smoothed_dist[j- (newtotal_shift)]) / 300
           
            if (j >= step_startcontrol):
            #calculate desired velocity
                desired_vel = (desired_distance - smoothed_dist[j]) / interva_p
         
            #calculate model contro output
           
                new_model_control = (desired_vel + vel_devi - linearparameter_vector[3]) /linearparameter_vector[2]
            
             
          
                
                
            else:
                if (j > total_shift):
                    new_model_control = (control_data[j] + vel_devi - linearparameter_vector[3]) /linearparameter_vector[2]
                else:
                    new_model_control = (control_data[j]  - linearparameter_vector[3]) /linearparameter_vector[2]
                    
        # parameters update vias SGD         
         #   if (j < step_startcontrol):
            parameter_vector,error,model_regout = self.SGD_optimizer(learning_rate,batchsize,local_vel,parameter_vector,current_control)
    
        # parameters update vias SGD     
        
            linearparameter_vector,linearerror,linearmodel_regout = self.LinearSGD_optimizer(learning_rate,batchsize,local_vel,linearparameter_vector,current_control)
        
        #calculate numeric modelled controller output from polynomial model
      #  print(parameter_vector)
            if (j > total_shift):
                if (j < step_startcontrol):
                    polynomial_model_control = self.newton_control_solver(polynomial_model_control,current_control,parameter_vector)
                else:
                     polynomial_model_control = self.newton_control_solver(polynomial_model_control,desired_vel+ vel_devi,parameter_vector)
                      #calculate simulated noisy movement of piston from generated controller output
            
                     noisy_movement = self.movement_piston_simulation(interva_p,current_control_array,polynomial_model_control,parameter_vector,system_SNR)
            #update position of piston
                     position_update = measured_dist[j] + noisy_movement
                     if (j+1 < iteration-batchsize ):
                        measured_dist[j+1] = measured_dist[j] + noisy_movement
                        #calculate real smoothed distance with linear regression
                        
                        m,b = self.estimate_coef(np.arange(interval_smooth_reg),measured_dist[j-(interval_smooth_reg-1):j+1] )
                      
                        smoothed_dist[j+1] = b
                  #   print(smoothed_dist[j],position_update,polynomial_model_control)
            else:
                polynomial_model_control = local_vel
                          
                
            polynomial_model_regout_array.append(polynomial_model_control)
        
        
            
        #array append
            w1_array.append(parameter_vector[0])
            w2_array.append(parameter_vector[1])
            w3_array.append(parameter_vector[2])
            w4_array.append(parameter_vector[3])
       
            error_array.append(error)
            new_model_controlarray.append(new_model_control)
            predicted_distancearray.append(predicted_distance)
            measured_distarray.append(measured_dist[j])
            current_control_array.append(current_control)
        
    
            vel_array.append(local_vel)
            desired_distance_array.append(desired_distance)
            model_regout_array.append(linearmodel_regout)
         
    #plot predicted and smoothed distance, desired distance
        total_shift = 26
    
        plt.plot(np.arange(iteration)[100::],np.array(predicted_distancearray)[100::].flatten())
        plt.plot(np.arange(iteration)[100:int(0.8*iteration)],np.array(predicted_distancearray)[100+total_shift:int(0.8*iteration)+total_shift].flatten())
        plt.plot(np.arange(iteration),np.array(real_distancearray).flatten())
        plt.plot(np.arange(iteration),np.array(measured_distarray).flatten())
        plt.plot(np.arange(iteration),np.array(desired_distance_array).flatten())
        plt.legend(['predicted distance','shifted prediction','smoothed','measured','desired'])
        plt.title('Distance graph with polynomial model')
        plt.show()
    #plot modelled control output and current control output
    
        plt.plot(np.arange(iteration-batchsize)[20::],np.array(current_control_array).flatten()[20::])
        plt.plot(np.arange(iteration-batchsize)[20::],np.array(model_regout_array).flatten()[20::])
        plt.title('The comparison between real and fitted velocity with polynomial model')
        plt.xlabel('iteration')
        plt.legend(['real','model'])
        plt.show()
    
        plt.plot(np.arange(iteration-batchsize)[total_shift::],(32768+3276.8*np.array(vel_array).flatten()[total_shift::]))
   
        plt.plot(np.arange(iteration-batchsize)[total_shift::],(32768+3276.8*np.array(new_model_controlarray).flatten()[total_shift::]))
        plt.plot(np.arange(iteration-batchsize)[total_shift::],(32768+3276.8*np.array(polynomial_model_regout_array).flatten()[total_shift::]))
        plt.xlabel('iteration')
        plt.legend(['real','linear model','polynomial model'])
        plt.title('The comparison between real and estimated control output after starting model control')
        plt.show()
    
    #plot shifted control output vs velocity data
        plt.scatter(np.array(current_control_array)[0::].flatten(),np.array(vel_array)[0::].flatten())
        plt.scatter(np.array(model_regout_array)[0::].flatten(),np.array(vel_array)[0::].flatten())
        plt.xlabel('iteration')
        plt.legend(['real','model'])
        plt.show()
    
    #plot parameter update
        plt.plot(np.arange(iteration),np.array(w1_array).flatten())
        plt.plot(np.arange(iteration-batchsize),np.array(w2_array).flatten())
        plt.plot(np.arange(iteration),np.array(w3_array).flatten())
        plt.plot(np.arange(iteration),np.array(w4_array).flatten())
        plt.xlabel('iteration')
        plt.legend(['$w_1$','$w_2$','$w_3$','$w_4$'])
        plt.show()
    #plot error
    
        plt.plot(np.arange(iteration)[0:],np.array(error_array).flatten()[0:])
        plt.xlabel('iteration')
        plt.legend(['normalized error'])
        plt.show()
if  __name__ == '__main__':
 
    #experiment datapath
        datapath =  r'C:\Users\zchen\Downloads\data (97).dat'  
    # read data from sensor measurement DAT. file
        raw_measurementdata = pd.read_table(datapath,skiprows = 6, delimiter='\t',engine='python')
    #construct the class
        R = Model_control_SGD(raw_measurementdata)
    
    #start training with data
        R.data_fitting_model_main(R.interval_smooth_reg,R.total_shift,R.measured_dist,R.system_SNR,R.interva_p,R.counter,R.step_startcontrol,R.op_amplitude,R.op_freq, R.sampling_freq, R.op_back_vel, R.predicted_distance,R.batchsize,R.initial_w,R.iteration,R.learning_rate,R.control_data,R.velocity_data,R.smoothed_dist)
    
    

    
    
    
    
    
   
