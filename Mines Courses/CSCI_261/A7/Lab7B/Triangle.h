

#pragma once



class Triangle {
    public:
        Triangle();        // Photo constructor function
        double side1;
        double side2;
        double side3;
        double perimeter;
        double area;
        double getSideOne();
        double getSideTwo();
        double getSideThree();
        void setSideOne(double s1);
        void setSideTwo(double s2);
        void setSideThree(double s3);
        bool validate();
        double findArea();
        double findPerimeter();
        bool isLarger(double userArea);
        
    private:    
        
        double _sideOne;
        double _sideTwo;
        double _sideThree;
        
    
};

void getTriangleInfo();