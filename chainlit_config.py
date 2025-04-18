import chainlit as cl

@cl.on_start
def on_start():
    if cl.theme == "dark":
        cl.set_logo("C:\MediMateAI\public\logo_dark.png")
    else:
        cl.set_logo("C:\MediMateAI\public\logo_light.png")
