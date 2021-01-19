#include "IntervalTree.h"

int IntervalTree::Median(std::vector<int> &p) {
    std::sort(p.begin(), p.end());
    //printVector(p);
    if (p.size() % 2 == 0)
        return (p[p.size() / 2 - 1] + p[p.size() / 2]) / 2;
    else
        return p[p.size() / 2];
}

void IntervalTree::build(std::vector<Interval> &i) {
    buildTree(root,&i);
}

void IntervalTree::buildTree(ITreeNode* n, std::vector<Interval>* intervals) {

    if(n == nullptr) {
        return;
    }

    //get the middle points such that it split all intervals in half
    std::vector<int> endpoints;
    for(auto &element: *intervals) {
        endpoints.push_back(element.start);
        endpoints.push_back(element.end);
    }
    int mid = Median(endpoints);
    n->centerPoint = mid;
    //printVector(intervals);
    //printf("mid point: %d\n",mid);

    std::vector<Interval> left,right,intersect;
    for(auto &element : *intervals) {
        if(element.end <= mid) {
            left.push_back(element);
        }
        else if(element.start > mid) {
            right.push_back(element);
        }
        else {

//            if(element.start > mid) {
//                printf("[%d,%d],%d\n",element.start,element.end,mid);
//                right.push_back(element);
//                continue;
//            }
//            if(element.end <= mid) {
//                printf("[%d,%d],%d\n",element.start,element.end,mid);
//                left.push_back(element);
//                continue;
//            }
            intersect.push_back(element);
        }
    }
//    printf("===left===\n");
//    printVector(&left);
//    printf("===right===\n");
//    printVector(&right);



    std::sort(intersect.begin(),intersect.end(),
              [](const Interval& a, const Interval& b) -> bool {
                  return a.start < b.start;
              });
    for(auto &element: intersect)
        n->lIntervalList.push_back(element);

    std::sort(intersect.begin(),intersect.end(),
               [](const Interval& a, const Interval& b) -> bool {
                   return a.end > b.end;
               });
    for(auto &element: intersect)
        n->rIntervalList.push_back(element);

//    printf("===intersect===\n");
//    printVector(&intersect);


//    ITreeNode lnode,rnode;
//    n->lchild = &lnode;
//    n->rchild = &rnode;

    if(!right.empty()) {
        n->rchild = new ITreeNode;
    }
    if(!left.empty()) {
        n->lchild = new ITreeNode;
    }


    buildTree(n->lchild,&left);
    buildTree(n->rchild,&right);

}

void IntervalTree::stabbingQuery(int x, int& counter) {
    stab(root,x,counter);
}

void IntervalTree::stab(ITreeNode *n, int x, int& counter) {
    if(n == nullptr)
        return;
    if(x >= n->centerPoint) {
        for(auto &element: n->rIntervalList) {
            if(element.end >= x)
                counter++;
            else
                break;
        }
        stab(n->rchild,x,counter);
    }
    else {
        for(auto &element: n->lIntervalList) {
            if(element.start <= x)
                counter++;
            else
                break;
        }
        stab(n->lchild,x,counter);
    }
}

void IntervalTree::insert(Interval& i) {
   _insert(root, i);
}

void IntervalTree::_insert(ITreeNode* n, Interval& i) {

}


void IntervalTree::deleteTree(ITreeNode* r) {
    if(r == nullptr)
        return;

    deleteTree(r->lchild);
    deleteTree(r->rchild);
    delete r;
}
