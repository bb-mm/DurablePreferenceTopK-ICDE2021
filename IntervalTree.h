#ifndef INTERVALTREE_H
#define INTERVALTREE_H

#include <iostream>
#include <vector>
#include <assert.h>
#include <algorithm>

struct Interval {
    //intervals are represented as [start,end]
    Interval(int s, int e): start(s), end(e) {}

    int start; //left-endpoint of interval
    int end; //right-endpoint of interval

    bool intersects(Interval &i) {
        if(start <= i.start && end >= i.end) //interval i is enclosed by the given interval
            return true;
        else if (start >= i.start && end <= i.end) //interval i encloses the given interval
            return true;
        else if (start <= i.start && start >= i.end) //interval i overlaps the given interval
            return true;
        else if (end >= i.start && end <= i.end) //interval i overlaps the given interval
            return true;
        else { //interval i doesn't overlap with the given interval
            assert(start > i.end || end < i.start);
            return false;
        }
    }

    bool contains(int t) {
        return t>=start && t <= end;
    }
    void printInterval() {
        //printf("[%d,%d)\n",start,end);
    }
};

struct ITreeNode {
    ITreeNode() {
        lchild = nullptr;
        rchild = nullptr;
        centerPoint = 0;
    }
    ~ITreeNode() {}

    int centerPoint;
    std::vector<Interval> lIntervalList; //intervals stored at the node sorting by left endpoint
    std::vector<Interval> rIntervalList; //intervals stored at the node sorting by right endpoint
    ITreeNode* lchild;
    ITreeNode* rchild;

};

class IntervalTree {
public:
    IntervalTree() {
        std::cout << "creating an interval tree instance..." << std::endl;
        root = new ITreeNode;
    }
    ~IntervalTree() {
        std::cout << "destorying an interval tree instance..." << std::endl;
        deleteTree(root);
    }

    void build(std::vector<Interval>& i);

    void stabbingQuery(int x, int& counter); //stabbing query with a point

    void insert(Interval& i);



private:

    void buildTree(ITreeNode* root, std::vector<Interval>* intervals);

    void _insert(ITreeNode* root, Interval& i);

    void stab(ITreeNode* n, int x, int& counter); //for public method stabbingQuery()

    void deleteTree(ITreeNode* r);

    int Median(std::vector<int> &p);

    ITreeNode* root;

    //int totalIntervals;
    std::vector<Interval> allIntervals;

};

#endif //DTOPK_INTERVALTREE_H
