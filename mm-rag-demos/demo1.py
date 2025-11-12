from gradio_utils import get_demo


debug = False # change this to True if you want to debug

demo = get_demo()
demo.launch(server_name="0.0.0.0", server_port=9999, debug=debug, share=True)

