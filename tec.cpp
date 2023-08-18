#include <iostream>

class A
{
public:
    A(int x)
    {
        std::cout << "A's constructor called with x = " << x << std::endl;
    }
};

class B
{
public:
    B(int y)
    {
        std::cout << "B's constructor called with y = " << y << std::endl;
    }
};

class C
{
public:
    C(int z)
    {
        std::cout << "C's constructor called with z = " << z << std::endl;
    }
};

class Derived : public A
{
public:
    Derived(int x, int y, int z) : A(x), innerB(y), innerC(z)
    {
        std::cout << "Derived's constructor called" << std::endl;
    }

private:
    C innerC;
    B innerB;
};

int main()
{
    Derived d(1, 2, 3);
    return 0;
}
