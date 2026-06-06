import zipfile

with open("/tmp/test.zip", "wb") as f:
  with zipfile.ZipFile(f, "w") as z:
    z.writestr("test.txt", "hello")

with open("/tmp/test.zip", "rb") as f:
  with zipfile.ZipFile(f, "r") as z:
    print(z.read("test.txt"))
