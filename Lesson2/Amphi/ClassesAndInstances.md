
# Amphi 2 - Classes and Instances

# 1. Introduction

Basic types in Python are numbers, string, iterable (list, tuple, set, dictionary). They can be used to describe a variety of objects in reality.

Classes in Python allow users to define their own type.

For example, we would like to describe some apartments with their properties: area, type, rooms, location, construction year. We can use dictionary in Python like this:


```python
apart1 = {"area": 45, "type": "T2", "rooms": ["living room", "bedroom1"], "location": "91120", "construction year": 2001}
apart2 = {"area": 24, "type": "studio", "rooms": ["living room"], "location": "31300", "construction year": 1976}
```

In code edition, using basic type to describe complex objects may be inconvenient (to be considered in the following parts). For example, we cannot distinguish a dict of type "apartment" with other types.


```python
type(apart1)
```




    dict



We may need to create a function "manually" to check if some object is of type appartment, like this:


```python
def isApartment(obj):
    return set(obj.keys()) == {"area", "type", "rooms", "location", "construction year"}
isApartment(apart1)
```




    True



In Python, we can use **class** to describe a new type. For the same problem, we can create a class **Apartment** like this.


```python
class Apartment:
    def __init__(self, area, apart_type, rooms, location, construction_year):
        self.area = area
        self.apart_type = apart_type
        self.rooms = rooms
        self.location = location
        self.construction_year = construction_year
    
    def __str__(self):
        return "Appartment(%d, %s, %s, %s, %d)" % (self.area, self.apart_type, self.rooms, self.location, self.construction_year)

Apart1 = Apartment(45, "T2", ["living room, bedroom1"], "91120", 2001)
Apart2 = Apartment(24, "studio", ["living room"], "31300", 1976)
print(Apart1)
print(Apart2)
```

    Appartment(45, T2, ['living room, bedroom1'], 91120, 2001)
    Appartment(24, studio, ['living room'], 31300, 1976)
    

In the above example:
- We have created the class Apartment to describe objects of type apartment.
- We have specified (in the \__init\__ function) that properties we are interested in are area, type of apartment, rooms, location and year of construction.
- We have specified (in the \__str\__ function) how the objects of this class **Apartment** should be printed.


- Then, we have defined two objects **Apart1** and **Apart2** of type **Apartment**.

## 1.1 Classes and instances

The above example have shown us how to define a class. Syntax for class definition is as follows:
class ClassName:
    <statement-1>
    .
    .
    .
    <statement-N>
A class can be seen as a new type defined by users. Objects of a class are called **instances** of the class.

For example, **Apart1** and **Apart2** are two instances of the class **Apart**. To check an object is an instance of some class, use **\__class\__** or **isInstance**.


```python
print(type(Apart1)) #In Python 2, type 'instance'; in Python 3, __main__.Apartment
print(Apart1.__class__)
print(isinstance(Apart1, Apartment))
```

    <type 'instance'>
    __main__.Apartment
    True
    

## 1.2 Instantiation

The function **\__init\__** defined inside the class definition is used to construct new instances of a class. In the above example, we want to specify that an instance of class **Apartment**, the so-call "**self**" here, will be defined by 5 arguments: area, apart_type, rooms, location and construction_year. 

At instance's creation, the **\__init\__** function will be called and assign values of these 5 arguments **in correct order** to the 5 corresponding attributes of **self**.


```python
class Apartment:
    def __init__(self, area, apart_type, rooms, location, construction_year):
        self.area = area
        self.apart_type = apart_type
        self.rooms = rooms
        self.location = location
        self.construction_year = construction_year

Apart1 = Apartment(45, "T2", ["living room, bedroom1"], "91120", 2001)
```

We can now refind attributes of the instance.


```python
print(Apart1.area)
```

    45
    

We can also specify the keyword of parameter in instance creation to make the code clear.


```python
Apart1 = Apartment(apart_type = "T2", rooms = ["living room, bedroom1"], construction_year = 2001, location = "91120", area = 45)
print(Apart1.area)
```

    45
    

We can also define a default value for all instances of a class by specifying it in the parameters of **\__init\__**. Attention, in this case, in **\__init__** function, "default" arguments must follow "non default" arguments.


```python
class Apartment:
    # The following syntax should raise an error.
    # def __init__(self, area = 20, apart_type, rooms, location, construction_year = 2000):
    
    def __init__(self, apart_type, rooms, location, area = 20, construction_year = 2000):
        self.area = area
        self.apart_type = apart_type
        self.rooms = rooms
        self.location = location
        self.construction_year = construction_year

Apart1 = Apartment(apart_type = "T2", rooms = ["living room, bedroom1"], location = "91120", area = 45)
print(Apart1.area, Apart1.apart_type, Apart1.construction_year)
```

    (45, 'T2', 2000)
    

## 1.3 Instance attributes

In the above example, **self.area**, **self.apart_type**, ... are **instance variables** or **instance attributes**. It is properties of an instance and not the whole class. It is in general different for different instances of the class.


```python
Apart1 = Apartment(apart_type = "T2", rooms = ["living room, bedroom1"], location = "91120", area = 45)
Apart2 = Apartment(apart_type = "studio", rooms = ["living room"], location = "31300", area = 24, construction_year = 1976)
print(Apart1.area, Apart2.area)
```

    (45, 24)
    

## 1.4 Instance methods

In Python 2, any function defined in the body of a class definition is by default an instance method, that means, the method can be called by instances of the class. However, in order that an instance is able to call that method, the keyword **self** must be specified as the first argument of a function.

In Python 3, any function defined in the body of a class definition that has **self** as the first argument is considered as an instance method.


```python
class Apartment:    
    def __init__(self, apart_type, rooms, location, area = 20, construction_year = 2000):
        self.area = area
        self.apart_type = apart_type
        self.rooms = rooms
        self.location = location
        self.construction_year = construction_year
    
    def sayHello():
        #An instance method in Python2, not an instance method in Python 3.
        print("Hello!")
    
    def sayHello2(self):
        print("Hello!")

Apart1 = Apartment("T2", ["living room, bedroom1"], "91120", 45)
        
print(type(Apartment.sayHello)) #class function in Python 3
#Apart1.sayHello() #Should raise error
#Apartment.sayHello(Apart1) #Should raise error
#Apartment.sayHello() #OK in Python 3 but raise error in Python 2.

print(type(Apartment.sayHello2)) #class function in Python 3
Apartment.sayHello2(Apart1)
# This way is more popular
Apart1.sayHello2()

```

    <type 'instancemethod'>
    <type 'instancemethod'>
    Hello!
    Hello!
    

Instance methods are very useful to access and handle attributes of instances.


```python
class Apartment:    
    def __init__(self, apart_type, rooms, location, area = 20, construction_year = 2000):
        self.area = area
        self.apart_type = apart_type
        self.rooms = rooms
        self.location = location
        self.construction_year = construction_year
    
    def getNumberOfRooms(self):
        return len(self.rooms)

Apart2 = Apartment(apart_type = "studio", rooms = ["living room"], location = "31300", area = 24, construction_year = 1976)
print(Apart2.getNumberOfRooms())
```

    1
    

## 1.5 Convention for "private" attributes

In Python, if an attribute of an instance of a class has the name beginning with **\_\_** (underscore), its value cannot be accessed outside the class by calling instance.attribute.


```python
class Apartment:    
    def __init__(self, apart_type, rooms, location, area = 20, construction_year = 2000, price = 100000):
        self.area = area
        self.apart_type = apart_type
        self.rooms = rooms
        self.location = location
        self.construction_year = construction_year
        self.__price = price

Apart1 = Apartment("T2", ["living room, bedroom1"], "91120", 45)
#print(Apart1.__price) #Should raise an error
print(Apart1.location)
#But
print(Apart1._Apartment__price)
```

    91120
    100000
    

This can be used to restrain manipulation of stable variables and does not mean the variable is totally private. In Python, privacy is a compiler level concept.

## 1.6 Getters and setters

Usually people declare all attributes of instances of class as "private", then write getters and setters to allow other programs/users access or modify attributes. In general, we use the term **getters** to describe instance methods that get values of one or some attributes of instances (and do nothing else). We use **setters** to describe instance methods that modify values of one or some attributes.


```python
class RectangleApartment:
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__area = length * width
    
    def getLength(self):
        return self.__length

    def getWidth(self):
        return self.__width
    
    def getArea(self):
        return self.__area
    
    def setLength(self, newlength):
        self.__length = newlength
        self.__area = self.__length * self.__width
    
    def setWidth(self, newwidth):
        self.__width = newwidth
        self.__area = self.__length * self.__width

A = RectangleApartment(15, 5)
print(A.getArea())
A.setLength(16)
print(A.getArea())
A.setWidth(6)
print(A.getArea())
```

    75
    80
    96
    

## 1.7 String representation of an instance

Without a declaration of the **\_\_str\_\_** method, printing an instance gives a result like this:


```python
class RectangleApartment:
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__area = length * width

A = RectangleApartment(15, 5)
print(A)
```

    <__main__.RectangleApartment instance at 0x000000000354D948>
    

We can define how such an instance would be printed. An instance where a **\_\_str\_\_** method has been defined can be converted into **str**.


```python
class RectangleApartment:
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__area = length * width
    
    def __str__(self):
        return "RectangleAppartment(length = %d, width = %d, area = %d)" % (self.__length, self.__width, self.__area)

A = RectangleApartment(15, 5)
print(A)
str(A)
```

    RectangleAppartment(length = 15, width = 5, area = 75)
    




    'RectangleAppartment(length = 15, width = 5, area = 75)'



## 1.8 Make an instance iterable

Like **str**, **list**, **tuple**, **set**, **dict**, we can define a customized iteration for an instance by using **\_\_iter\_\_** and **\_\_next\_\_**/**next**. With these methods, we can convert an instance into **list**, **tuple**, **set**.


```python
class Counter:
    def __init__(self, low, high):
        self.current = low
        self.high = high

    def __iter__(self):
        return self

    def next(self): # Python 2
#   def __next__(self): #Python 3
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.current - 1
        
C = Counter(5, 10)
for c in C:
    print c
    
C = Counter(5, 10)
print(list(C))

C = Counter(5, 10)
print(set(C))
```

    5
    6
    7
    8
    9
    10
    [5, 6, 7, 8, 9, 10]
    set([5, 6, 7, 8, 9, 10])
    

# 2. Class attributes, class methods, static methods

## 2.1 Class attributes

A class attribute is a variable of the class itself. If it is public (the name does not begin with **\_\_**), it can be accessed from all instances of the class.


```python
class RectangleApartment:
    number_apartments = 0
    
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__area = length * width
        RectangleApartment.number_apartments += 1

A1 = RectangleApartment(18, 5)
print(RectangleApartment.number_apartments)
print(A1.number_apartments)
A2 = RectangleApartment(20, 4)
print(RectangleApartment.number_apartments)
print(A1.number_apartments)
print(A2.number_apartments)
```

    1
    1
    2
    2
    2
    

It can be modified as a class variable


```python
class RectangleApartment:
    number_apartments = 0
    
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__area = length * width
        RectangleApartment.number_apartments += 1

RectangleApartment.number_apartments = 5
print(RectangleApartment.number_apartments)
A1 = RectangleApartment(18, 5)
A2 = RectangleApartment(20, 4)
print(A1.number_apartments)
```

    5
    7
    

Attention: an attempt **instance.class_attribute = some_value** will create an instance attribute having the same name, but not modify the class attribute.


```python
class RectangleApartment:
    number_apartments = 0
    
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__area = length * width
        RectangleApartment.number_apartments += 1

A1 = RectangleApartment(18, 5)
A2 = RectangleApartment(20, 4)
print(RectangleApartment.number_apartments)
print(A1.number_apartments)
print(A2.number_apartments)

A1.number_apartments = 10
print(RectangleApartment.number_apartments)
print(A1.number_apartments)
print(A2.number_apartments)
```

    2
    2
    2
    2
    10
    2
    

Therefore, if an instance attribute and a class attribute has the same name, calling that name from an instance will return the instance variable.

## 2.2 Class methods

By default, a function defined inside a class definition is considered as an instance method in Python 2 and a static method in Python 3.

A class method can be specified with
@classmethod

It can modified attributes of a class. The first argument of the class method will be the classname.


```python
class RectangleApartment:
    number_apartments = 0
    
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__area = length * width
        RectangleApartment.number_apartments += 1
    
    @classmethod
    def reset_number_apartments(cls):
        cls.number_apartments = 0

RectangleApartment.number_apartments = 5
print("Begin: ", RectangleApartment.number_apartments)
A1 = RectangleApartment(18, 5)
A2 = RectangleApartment(20, 4)
print("After defining 2 instances: ", RectangleApartment.number_apartments)
RectangleApartment.reset_number_apartments()
print("After reset: ", RectangleApartment.number_apartments)
```

    ('Begin: ', 5)
    ('After defining 2 instances: ', 7)
    ('After reset: ', 0)
    

**Attention:** Calling a class function from an instance will modify class attributes, and not only instance attributes.


```python
class RectangleApartment:
    number_apartments = 0
    
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__area = length * width
        RectangleApartment.number_apartments += 1
    
    @classmethod
    def reset_number_apartments(cls):
        cls.number_apartments = 0

print("Begin: ", RectangleApartment.number_apartments)
A1 = RectangleApartment(18, 5)
A1.number_apartments = 3
A2 = RectangleApartment(20, 4)
print("After defining 2 instances: ", RectangleApartment.number_apartments)
A1.reset_number_apartments()
print("After reset, class attribute: ", RectangleApartment.number_apartments)
print("After reset, instance attribute: ", A1.number_apartments)
```

    ('Begin: ', 0)
    ('After defining 2 instances: ', 2)
    ('After reset, class attribute: ', 0)
    ('After reset, instance attribute: ', 3)
    

## 2.3 Static methods

We can also define static methods in a class. Static methods are functions that acts neither on attributes of the current class nor on attributes of the current instance. We can specify a static method by specifying **@staticmethod** before the function definition in Python 2, or nothing in Python 3.


```python
class RectangleApartment:
    number_apartments = 0
    
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__area = length * width
        RectangleApartment.number_apartments += 1
    
    @staticmethod
    def add(a, b):
        return a + b

print(RectangleApartment.add(2, 3))
```

    5
    

In Python 3, we CANNOT call a static method from an instance.
In Python 2, we CAN call a static method from an instance. It does not modify instance nor class attributes.


```python
class RectangleApartment:
    number_apartments = 0
    
    def __init__(self, length, width):
        self.__length = length
        self.__width = width
        self.__area = length * width
        RectangleApartment.number_apartments += 1
    
    @staticmethod
    def add(a, b):
        return a + b

a = RectangleApartment(10, 5) 
print(a.add(2, 3)) #Should raise error in Python 3, use RectangleApartment.add(2, 3) instead.
```

    5
    

If a static method is not related to a class itself, we should define it outside a class definition.


```python
def add(a, b):
    return a + b
```

In this way, While importing a static function from a module, it suffices to use **from mymodule import add** instead of **from mymodule import SomeClass**.

# 3. Inheritance

## 3.1 Inheritance

Suppose we have two class **A** and **B**, and relation that every instance of class **B** is also an instance of class **A**. We say that **B** is derived from **A**. Instead of copy all attributes and methods from **A**'s definition to **B**, we can use **class B(A)**.

To init a derived class, we can first derive its parent and then add updates. We can also rewrite methods in derived class by modifying methods in parent's class.


```python
class Apartment:
    def __init__(self, area, apart_type, rooms, location, construction_year):
        self.area = area
        self.apart_type = apart_type
        self.rooms = rooms
        self.location = location
        self.construction_year = construction_year
    
    def __str__(self):
        return "Appartment(%d, %s, %s, %s, %d)" % (self.area, self.apart_type, self.rooms, self.location, self.construction_year)

class RectangleApartment(Apartment):    
    def __init__(self, apart_type, rooms, location, construction_year, length, width):
        Apartment.__init__(self, length*width, apart_type, rooms, location, construction_year)
        self.length = length
        self.width = width
    
    def __str__(self):
        return "Appartment(%d, %s, %s, %s, %d, %d, %d)" % (self.area, self.apart_type, self.rooms, \
                                                           self.location, self.construction_year, self.length, self.width)

A = RectangleApartment("T2", ["Living room", "Bedroom1"], "91120", 2001, 20, 5)
print(A.area)
print(A)
```

    100
    Appartment(100, T2, ['Living room', 'Bedroom1'], 91120, 2001, 20, 5)
    

If a method was not defined in the derived class, the method in parent class will be called.


```python
class Apartment:
    def __init__(self, area, apart_type, rooms, location, construction_year):
        self.area = area
        self.apart_type = apart_type
        self.rooms = rooms
        self.location = location
        self.construction_year = construction_year
    
    def __str__(self):
        return "Appartment(%d, %s, %s, %s, %d)" % (self.area, self.apart_type, self.rooms, self.location, self.construction_year)

class RectangleApartment(Apartment):    
    def __init__(self, apart_type, rooms, location, construction_year, length, width):
        Apartment.__init__(self, length*width, apart_type, rooms, location, construction_year)
        self.length = length
        self.width = width

A = RectangleApartment("T2", ["Living room", "Bedroom1"], "91120", 2001, 20, 5)
print(A.area)
print(A)
```

    100
    Appartment(100, T2, ['Living room', 'Bedroom1'], 91120, 2001)
    

## 3.2 Multiple inheritance

A class can be derived from multiple parent classes, using **class B(A1, A2, ..., An)**. If a method is called and is not defined in **B**, then the method in **A1** will be called if it exists, then **A2**, **A3**, ...


```python
class Parallelogram():
    def __init__(self, side1, side2, angle):
        self.side1 = side1
        self.side2 = side2
        self.angle = angle
    def __str__(self):
        return "Parallelogram(%d, %d, %d)" % (self.side1, self.side2, self.angle)
    def getsides(self):
        return (self.side1, self.side2)

class Rectangle(Parallelogram):
    def __init__(self, side1, side2):
        Parallelogram.__init__(self, side1, side2, 90)
    def __str__(self):
        return "Rectangle(%d, %d)" % (self.side1, self.side2)

class Rhombus(Parallelogram):
    def __init__(self, side1, angle):
        Parallelogram.__init__(self, side1, side1, angle)

class Square(Rhombus, Rectangle):
    def __init__(self, side):
        Rhombus.__init__(self, side, 90)

A = Square(10)
print(isinstance(A, Square))
print(isinstance(A, Rectangle))
print(isinstance(A, Rhombus))
print(isinstance(A, Parallelogram))
print(A.getsides())
print(A)  #Should be rectangle if we comment out the __str__ method in Parallelogram.
```

    True
    True
    True
    True
    (10, 10)
    Parallelogram(10, 10, 90)
    

# 4. Exception


```python

```


```python

```
