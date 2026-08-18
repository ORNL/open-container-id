with open("src/container_id/streams/rtsp.py", "r") as f:
    content = f.read()

content = content.replace("logger.warning(\"Failed to close container\")", "pass")
with open("src/container_id/streams/rtsp.py", "w") as f:
    f.write(content)
