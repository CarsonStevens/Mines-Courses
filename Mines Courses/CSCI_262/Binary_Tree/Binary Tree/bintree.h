//head file with all the functions and classes of the program

#ifndef BINTREE_H
#define BINTREE_H
#include <stdlib.h>

template <class Item>
struct BinaryTreeNode
{
	Item data;
	BinaryTreeNode *left;
	BinaryTreeNode *right;

};

template <class Item>
BinaryTreeNode<Item>* create_node
(
 const Item& entry,
 BinaryTreeNode<Item>* l_ptr,
 BinaryTreeNode<Item>* r_ptr
 );

template <class Item>
bool is_leaf(const BinaryTreeNode<Item>& node);

template <class Item>
void tree_clear(BinaryTreeNode<Item>*& root_ptr);

template <class Item>
BinaryTreeNode<Item>* tree_copy(BinaryTreeNode<Item>* root_ptr);

#include "bintree.template"
#endif
