#!/usr/bin/env python
# coding: utf-8

# # Lesson 1: Interactive Demo
# 
# Welcome to Lesson 1!
# 
# To access the `requirements.txt`, the `gradio_utils.py`and all needed helpers files for this course, go to `File` and click `Open`.
# 

# In[2]:


from gradio_utils import get_demo


# <p style="background-color:#fff1d7; padding:15px; "> <b>Note:</b>
# Take a minute to review the code in `gradio_utils`. 
# <br>
# We will cover various components used here in subsequent lessons.
# </p>

# In[ ]:


#You will need to restart the kernel each time you rerun this cell;
#otherwise, the port will not be available.

debug = False # change this to True if you want to debug

demo = get_demo()
demo.launch(server_name="0.0.0.0", server_port=9999, debug=debug, share=True)


# <p style="background-color:#fff1d7; padding:15px; "> <b>Note:</b>
# <br>
# Here is the YouTube video link used for this demo: https://www.youtube.com/watch?v=7Hcg-rLYwdM
# </p>
# 

# To access the `gradio_utils.py`and all needed helpers files for this course, go to `File` and click `Open`.

# In[ ]:




