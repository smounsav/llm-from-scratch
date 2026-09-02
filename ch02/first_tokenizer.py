import re
text = "Hello, world. Is this-- test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
print(result)