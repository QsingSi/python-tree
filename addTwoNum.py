import errno


class Student:

    def __init__(self, score=0, age=20):
        self.__score = score
        self.__age = age

    def set_age(self, value):
        self.__age = value

    @property
    def score(self):
        return self.__score

    @score.setter
    def score(self, value):
        if value > 100 or value < 0:
            raise ValueError('a fuck score!!!')
        self.__score = value

    def __str__(self):
        return 'a Student object(score: %d age: %d)' % (self.__score, self.__age)
    __repr__ = __str__


class Phd(Student):
    def __init__(self, score=0, age=20, gender=1):
        super(Phd, self).__init__(score=score, age=age)
        self.__gender = gender

    @property
    def gender(self):
        return self.__gender

    @gender.setter
    def gender(self, value):
        assert value == 0 or value == 1
        self.__gender = value
