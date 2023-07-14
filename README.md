# Mixture of Diffusers Controlnet version

![2022-10-12 15_35_27 305133_A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, s_640x640_schelms_seed7178915308_gc8_steps50](https://user-images.githubusercontent.com/9654655/195362341-bc7766c2-f5c6-40f2-b457-59277aa11027.png)




# Feature
I implemented this repo in March when MixtureofDiffusers first released. Using the examples in the test.ipynb, you can generate images of unlimited pixel size on personal GPU devices like 3060 and 3090. 

With MixtureofDiffusers Controlnet version, you can generate an unlimited size image region by region and can use different condition to control each region. (performing (Multi)Controlnet on each region)   
  
          
             

# COMING SOON
- [x] add functions which is suitable for different width and height of region
- [x] support id controlnet scale in paras
- [ ] integrate tomesd to accelerate speed of generation 
- [ ]  resize image so that its width and height the same as region  
  
    
      
# REFERENCE
The code is basically modified on [mixture-of-diffusers](https://github.com/albarji/mixture-of-diffusers)
