#include <iostream>
using namespace std;

class List {
    friend ListIter;
    public:
    // ... as seen before
    protected:
        ListCell* First;
};

class ListIter {
    public:
        // default constructor
        ListIter() { ptr = NULL; };

        // constructor that sets iterator to point to a list
        ListIter(List& l) (ptr l.First;);

        // reset iterator to point to another list
        void reset(List& l) (ptr = l.First;);

        // return item currently pointed to by iterator
        char operator() ();

        // pointer iterator to next item in list
        void operator ++();

        // change the data item in the node pointed to by the iterator
        void operator = (char c) { ptr->contents = c; };

        // return TRUE if more items remain in the list being processed, FALSE otherwise
        int operator !(){ return ptr != NULL; };

    protected:
        ListCell* ptr;
}

char ListIter::operator()()
{
    if (ptr != NULL)
        return ptr->GetContents();
        else
        return NULL;
};

void ListIter::operator ++()
{
    if (ptr != NULL) ptr = ptr->GetNext();
};

int main()
{
    List l;
    l.add('a');
    l.add('b');
    l.add('c');
    for (ListIter i(l); !i; ++i)
    cout << i();
    return 0;
}