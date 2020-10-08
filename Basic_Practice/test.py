class rect:
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height

    def getArea(self):
        return self.width * self.height

class sq(rect):
    def __init__(self, width=0):
        self.width = width
        self.height = width

r1=rect(10,12)
print(r1.getArea())

r2=sq(10)
print(r2.getArea())
