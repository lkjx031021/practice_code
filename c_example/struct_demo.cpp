#include <iostream>
#include <string>

using namespace std;

struct Student {
  string name;
  char alias[20];
  bool sex;
  int age;
};

typedef struct Teacher {
  char name[20];
  char subject[20];
  int age;
} T;

void printStruct(Teacher tt);
void printStruct(Student *st);
void printStruct(struct Student stuArr[], int size);

int main()
{
  Student s1 = {"张三", "zhangsan", true, 20};   
  Student s2 = {"李四", "lisi", 0, 22};

  struct Student stuArr[2] = { s1, s2};
  struct Student* p = stuArr;
  int len_ = sizeof(stuArr) / sizeof(stuArr[0]);
  cout << "指针p指向的地址: " << p << endl;
  cout << "结构体数组地址: " << &stuArr << endl;
  cout << "学生数量: " << len_ << endl;
  for (int i = 0; i < len_; i++){
    cout << "姓名: " << (p + i)->name << ", 别名: " << stuArr[i].alias 
         << ", 性别: " << (stuArr[i].sex ? "男" : "女") << ", 年龄: " << stuArr[i].age << endl;
  }
  Teacher t1 = {"王老师", "数学", 40};
  T t2 = {"秦老师", "数学分析", 45};
  printStruct(t1);
  printStruct(t2);
  printStruct(&s1);
  printStruct(&s2);
  printStruct(p, len_);
  printStruct(stuArr, len_);

  return 0;

}

void printStruct(Teacher tt)
{
  cout << "姓名: " << tt.name << ", 科目: " << tt.subject << ", 年龄: " << tt.age << endl;
}

void printStruct(Student *st)
{
  cout << "姓名: " << st->name << ", 别名: " << st->alias << ", 性别: " << (st->sex ? "男" : "女") << ", 年龄: " << st->age << endl;
}
void printStruct(struct Student *stuArr, int size)
{
  cout << "stuArr:" << stuArr << endl;
  cout << "stuArr:" << &stuArr << endl;
  cout << "------------------" << endl;
  for (int i = 0; i < size; i++) {
    cout << "姓名: " << stuArr[i].name << ", 别名: " << stuArr[i].alias << ", 性别: " << (stuArr[i].sex ? "男" : "女") << ", 年龄: " << stuArr[i].age << endl;
  }
}