#include <math.h>

class QMath{
  public:
    //Absoulte value math
    static double ABS(double num)
    {
      return fabs(num);  
    }

    //tanh function
    static double Tanh(double num)
    {
      // y = x / 1+|x| 
      return (num/(1.0+ABS(num)));
    }

    //sigmoid function
    static double Sigmoid(double num)
    {
      //tanh the number
      num = tanh(num);

      //set the number to 0 zero if tanh is less than 0
      return (num > 0) * num;
    }

    //Leaky Relu Function
    static double LRelu(double num)
    {
      //if num > 0; y = x; else y = 0.01*x
      return ((num < 0) * num * 0.01) + ((num >= 0) * num);
    } 

    //Relu
    static double Relu(double num)
    {
      // if x > 0; y = x; else y = 0
      return (num >= 0) * num;
    } 

};