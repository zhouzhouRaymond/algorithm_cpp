#include <algorithm>
#include <iostream>
#include <queue>
#include <ranges>
#include <stack>
#include <string_view>
#include <vector>

#include "dbg-macro/dbg.h"

// todo: 增加代码注释

/* 排序算法命名空间
 * 实现了：
 * 1. 选择排序
 * 2. 插入排序
 * 3. 希尔排序
 * 4. 归并排序 应用实例：逆序对
 * 5. 快速排序 应用实例：查找中位数、查找第i大的数字
 * 6. 堆排序
 * */
namespace tiny_sort {
/* 选择排序
 * 从小到大依次排序
 * 稳定排序
 * */
std::vector<int> select_sort(std::vector<int> num) {
  int num_size = static_cast<int>(num.size());
  // min is current changing position
  for (int min = 0; min < num_size; ++min) {
    int iter_min = min;
    // find the min_index from the unsorted part
    for (int iter = min; iter < num_size; ++iter)
      // 稳定排序的关键 >
      iter_min = num[iter_min] > num[iter] ? iter : iter_min;
    // change
    std::swap(num[min], num[iter_min]);
  }
  return num;
}

/* 插入排序
 * 依次排列未排序序列
 * 稳定排序
 * */
std::vector<int> insert_sort(std::vector<int> num) {
  int num_size = static_cast<int>(num.size());
  for (int iter = 1; iter < num_size; ++iter) {
    int iter_min = 0;
    // find the insert position
    // 稳定排序的关键 <=
    while (iter_min <= iter && num[iter_min] <= num[iter]) ++iter_min;
    // move for current position
    for (; iter_min < iter; ++iter_min) std::swap(num[iter_min], num[iter]);
  }
  return num;
}

/* 希尔排序
 *
 * 非稳定排序*/
std::vector<int> shell_sort(std::vector<int> num) {
  int h = 1, num_size = static_cast<int>(num.size());
  // 寻找最大的h
  while (h < num_size / 3) h = 3 * h + 1;
  for (; h >= 1; h /= 3)
    // h-sorted
    for (int iter = h; iter < num_size; ++iter)
      for (int iter_min = iter; iter_min >= h && num[iter_min] < num[iter_min - h]; iter_min -= h)
        // 当h=1时，类似插入排序。
        std::swap(num[iter_min], num[iter_min - h]);

  return num;
}

/* 归并排序：从高到低
 * 输入参数范围[low, high)
 *
 * 非稳定排序*/
void _merge_high_low(std::vector<int>& num, int low, int high) {
  if (low >= (high - 1)) return;
  int middle = low + (high - low) / 2;
  _merge_high_low(num, low, middle);
  _merge_high_low(num, middle, high);
  // merge into num
  std::inplace_merge(num.begin() + low, num.begin() + middle, num.begin() + high);
}

std::vector<int> merge_high_low(std::vector<int> num) {
  _merge_high_low(num, 0, static_cast<int>(num.size()));
  return num;
}

/* 归并排序：从低到高
 * */
std::vector<int> merge_low_high(std::vector<int> num) {
  int nums_size = static_cast<int>(num.size());
  for (int sz = 2; sz <= nums_size; sz *= 2)
    for (int iter = 0; iter + sz - 1 < nums_size; iter += sz)
      std::inplace_merge(num.begin() + iter, num.begin() + iter + sz / 2 - 1,
                         num.begin() + iter + sz - 1);

  return insert_sort(num);
}

int _inverse_pairs(std::vector<int>& num, int low, int high) {
  if (low >= (high - 1)) return 0;
  int ret = 0, middle = low + (high - low) / 2;

  ret += _inverse_pairs(num, low, middle);
  ret += _inverse_pairs(num, middle, high);

  for (int left = middle - 1, right = high - 1; left >= low && high >= middle;) {
    if (num[left] > num[right]) {
      ret += right - middle + 1;
      --left;
    } else if (num[left] < num[right])
      --right;
    else
      break;
  }
  std::inplace_merge(num.begin() + low, num.begin() + middle, num.begin() + high);
  return ret;
}

int inverse_pairs(std::vector<int> num) {
  return _inverse_pairs(num, 0, static_cast<int>(num.size()));
}

int _inverse_pairs_important(std::vector<int>& num, int low, int high) {
  if (low >= (high - 1)) return 0;
  int ret = 0, middle = (high - low) / 2 + low;

  ret += _inverse_pairs_important(num, low, middle);
  ret += _inverse_pairs_important(num, middle, high);

  // 统计重要逆序对
  for (int iter_low = low, iter_high = middle; iter_low < middle; ++iter_low) {
    while (iter_high < high && num[iter_low] > num[iter_high] * 2LL) ++iter_high;
    ret += iter_high - middle;
  }
  std::inplace_merge(num.begin() + low, num.begin() + middle, num.begin() + high);
  return ret;
}

int inverse_pairs_important(std::vector<int> num) {
  return _inverse_pairs_important(num, 0, static_cast<int>(num.size()));
}

int _partition(std::vector<int>& num, int low, int high) {
  int iter_low = low, iter_high = high - 1;
  while (iter_low != iter_high) {
    while (num[iter_low] <= num[low] && iter_low < high) ++iter_low;
    while (num[iter_high] >= num[low] && iter_high > low) --iter_high;
    if (iter_low > iter_high) break;
    std::swap(num[iter_low], num[iter_high]);
  }
  std::swap(num[low], num[iter_high]);
  return iter_high;
}

void _quick_sort(std::vector<int>& num, int low, int high) {
  if (low >= (high - 1)) return;

  int partition = _partition(num, low, high);
  _quick_sort(num, low, partition);
  _quick_sort(num, partition + 1, high);
}

std::vector<int> quick_sort(std::vector<int> num) {
  _quick_sort(num, 0, static_cast<int>(num.size()));
  return num;
}

int select_index(std::vector<int> num, int k) {
  k = k - 1;
  int low = 0, high = static_cast<int>(num.size());
  while (low < high) {
    int iter = _partition(num, low, high);
    if (iter == k)
      return num[k];
    else if (iter > k)
      high = iter;
    else
      low = iter + 1;
  }
  return num[k];
}

void _swim(std::vector<int>& num, int low, int high) {
  // only change the left item
  for (; high / 2 >= low && num[high / 2] < num[high]; high /= 2) {
    int iter_change = 0;
    if (high / 2 == 0)
      iter_change = high / 2 - 1;
    else
      iter_change = (high - 1) / 2;
    std::swap(num[iter_change], num[high]);
    high = iter_change;
  }
}

void _sink(std::vector<int>& num, int low, int high) {
  while (2 * low + 1 <= high) {
    int iter = 2 * low + 1;
    if (iter < high && num[iter] < num[iter + 1]) ++iter;
    if (num[low] > num[iter]) break;
    std::swap(num[low], num[iter]);
    low = iter;
  }
}

void heap_sort_swim(std::vector<int>& num) {
  int nums_size = static_cast<int>(num.size()) - 1;
  for (int iter = nums_size / 2; iter > 0; --iter) _sink(num, iter, nums_size);

  for (int iter = 0; iter < nums_size;) {
    std::swap(num[iter], num[nums_size]);
    _swim(num, ++iter, nums_size);
  }
}

void heap_sort_sink(std::vector<int>& num) {
  int num_size = static_cast<int>(num.size()) - 1;
  for (int iter = num_size / 2; iter > 0; --iter) _sink(num, iter, num_size);

  for (int iter = num_size; iter > 0;) {
    std::swap(num[iter], num[0]);
    _sink(num, 0, --iter);
  }
}

int middle_finder(const std::vector<int>& num) {
  int ret = 0;
  // 大顶堆
  std::priority_queue<int, std::vector<int>, std::less<>> max_heap;
  // 小顶堆
  std::priority_queue<int, std::vector<int>, std::greater<>> min_heap;

  for (auto item : num) {
    max_heap.push(item);
    min_heap.push(max_heap.top());
    max_heap.pop();
    if (max_heap.size() < min_heap.size()) {
      max_heap.push(min_heap.top());
      min_heap.pop();
    }
  }

  if (max_heap.size() == min_heap.size())
    ret = (max_heap.top() + min_heap.top()) / 2;
  else
    ret = max_heap.top();

  return ret;
}

namespace test {
void test_sort() {
  std::vector<int> num{9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::cout << tiny_sort::middle_finder(num);
  //  auto tmp = tiny_sort::merge_high_to_low(num);
  //  auto tmp = tiny_sort::merge_low_to_high(num);
  //  std::cout << tiny_sort::inverse_pairs_important(num);

  //  auto tmp = tiny_sort::quick_sort(num);
  //  for (auto nums : tmp) std::cout << nums << " ";
  //  std::cout << select_index(num, 1);
}
}  // namespace test
}  // namespace tiny_sort

namespace tiny_search {
// return the val index
int binary_search(const std::vector<int>& num, const int& val, int low, int high) {
  if (low > high) return -1;
  int middle = low + (high - low) / 2, cmp = num[middle] - val;

  if (cmp > 0)
    return binary_search(num, val, low, middle - 1);
  else if (cmp < 0)
    return binary_search(num, val, middle + 1, high);
  else
    return middle;
}

// return the val index
int binary_search_for(const std::vector<int>& num, const int& val) {
  int low = 0, high = static_cast<int>(num.size()) - 1;
  while (low <= high) {
    int middle = low + (high - low) / 2, cmp = num[middle] - val;
    if (cmp > 0)
      high = middle - 1;
    else if (cmp < 0)
      low = middle + 1;
    else
      return middle;
  }
  return -1;
}

namespace test {
void test_search() {
  std::vector<int> num{1, 2, 3, 4, 5, 6, 7};
  std::cout << binary_search(num, 7, 0, static_cast<int>(num.size()) - 1) << std::endl;
  std::cout << binary_search_for(num, 7) << std::endl;
}
}  // namespace test
}  // namespace tiny_search

namespace tiny_rb_tree {
enum _rb_tree_color { _red, _black };

class rb_avl_tree_base {
 public:
  int _val;
  rb_avl_tree_base* _parent;
  rb_avl_tree_base* _left;
  rb_avl_tree_base* _right;
  _rb_tree_color _color;

  explicit rb_avl_tree_base(int value = 0, rb_avl_tree_base* p = nullptr,
                            rb_avl_tree_base* l = nullptr, rb_avl_tree_base* r = nullptr,
                            _rb_tree_color c = _red)
      : _val(value), _parent(p), _left(l), _right(r), _color(c) {}
};

class rb_tree {
 public:
  rb_tree();
  explicit rb_tree(rb_avl_tree_base* root);

  void insert(rb_avl_tree_base* node);
  void insert(int val);
  void delete_node(rb_avl_tree_base* node);
  void delete_node(int val);

  rb_avl_tree_base* search(int value);

  void pre_order();
  void in_order();
  void post_order();

  void bfs();
  void dfs();

 private:
  void _left_rotate(rb_avl_tree_base* node);
  void _right_rotate(rb_avl_tree_base* node);

  void _insert_fixup(rb_avl_tree_base* node);
  void _delete_fixup(rb_avl_tree_base* node);
  void _delete_transplant(rb_avl_tree_base* node, rb_avl_tree_base* new_node);

  void _pre_order(rb_avl_tree_base* node) const;
  void _in_order(rb_avl_tree_base* node) const;
  void _post_order(rb_avl_tree_base* node) const;

 public:
 private:
  rb_avl_tree_base* _root;
};

rb_tree::rb_tree() : _root(nullptr) {}

rb_tree::rb_tree(rb_avl_tree_base* root) : _root(root) {}

void rb_tree::_left_rotate(rb_avl_tree_base* node) {
  auto tmp = node->_right;
  node->_right = tmp->_left;
  if (tmp->_left != nullptr) tmp->_left->_parent = node;
  tmp->_parent = node->_parent;
  if (node->_parent == nullptr)
    _root = tmp;
  else if (node == node->_parent->_left)
    node->_parent->_left = tmp;
  else
    node->_parent->_right = tmp;
  tmp->_left = node;
  node->_parent = tmp;
}

void rb_tree::_right_rotate(rb_avl_tree_base* node) {
  auto tmp = node->_left;
  node->_left = tmp->_right;
  if (tmp->_right != nullptr) tmp->_right->_parent = node;
  tmp->_parent = node->_parent;
  if (node->_parent == nullptr)
    _root = tmp;
  else if (node == node->_parent->_left)
    node->_parent->_left = tmp;
  else
    node->_parent->_right = tmp;
  tmp->_right = node;
  node->_parent = tmp;
}

void rb_tree::insert(int val) {
  auto new_node = new rb_avl_tree_base(val, nullptr, nullptr, nullptr, _rb_tree_color::_red);
  insert(new_node);
}

void rb_tree::insert(rb_avl_tree_base* node) {
  rb_avl_tree_base *iter = _root, *p = nullptr;

  // 寻找插入位置
  while (iter != nullptr) {
    p = iter;
    if (node->_val < iter->_val)
      iter = iter->_left;
    else
      iter = iter->_right;
  }
  node->_parent = p;
  if (p != nullptr) {
    if (node->_val < p->_val)
      p->_left = node;
    else
      p->_right = node;
  } else
    _root = node;
  node->_color = _rb_tree_color::_red;
  _insert_fixup(node);
}

void rb_tree::_insert_fixup(rb_avl_tree_base* node) {
  while (node->_parent != nullptr && node->_parent->_color == _rb_tree_color::_red) {
    if (node->_parent == node->_parent->_parent->_left) {
      auto tmp_right = node->_parent->_parent->_right;
      if (tmp_right != nullptr && tmp_right->_color == _rb_tree_color::_red) {
        node->_parent->_color = _rb_tree_color::_black;
        tmp_right->_color = _rb_tree_color::_black;
        node->_parent->_parent->_color = _rb_tree_color::_red;
        node = node->_parent->_parent;
      } else if (node == node->_parent->_right) {
        node = node->_parent;
        _left_rotate(node);
      } else {
        node->_parent->_color = _rb_tree_color::_black;
        node->_parent->_parent->_color = _rb_tree_color::_red;
        _right_rotate(node->_parent->_parent);
      }
    } else {  // the right part
      auto tmp_left = node->_parent->_parent->_left;
      if (tmp_left != nullptr && tmp_left->_color == _rb_tree_color::_red) {
        node->_parent->_color = _rb_tree_color::_black;
        tmp_left->_color = _rb_tree_color::_black;
        node->_parent->_parent->_color = _rb_tree_color::_red;
        node = node->_parent->_parent;
      } else if (node == node->_parent->_left) {
        node = node->_parent;
        _right_rotate(node);
      } else {
        node->_parent->_color = _rb_tree_color::_black;
        node->_parent->_parent->_color = _rb_tree_color::_red;
        _left_rotate(node->_parent->_parent);
      }
    }
  }
  _root->_color = _rb_tree_color::_black;
}

void rb_tree::delete_node(int val) { delete_node(search(val)); }

// todo:
void rb_tree::delete_node(rb_avl_tree_base* node) {
  if (node == nullptr) return;
}

// todo:
void rb_tree::_delete_fixup(rb_avl_tree_base* node) {}

void rb_tree::_delete_transplant(rb_avl_tree_base* node, rb_avl_tree_base* new_node) {
  if (node->_parent == nullptr)
    _root = new_node;
  else if (node == node->_parent->_left)
    node->_parent->_left = new_node;
  else
    node->_parent->_right = new_node;
  new_node->_parent = node->_parent;
}

rb_avl_tree_base* rb_tree::search(int value) {
  rb_avl_tree_base* iter = _root;
  while (iter != nullptr) {
    if (iter->_val == value)
      return iter;
    else if (iter->_val > value)
      iter = iter->_left;
    else
      iter = iter->_right;
  }
  return nullptr;
}

// todo: 非递归版本
// 输出格式 value | _color
void rb_tree::pre_order() { _pre_order(_root); }

void rb_tree::in_order() { _in_order(_root); }

void rb_tree::post_order() { _post_order(_root); }

void rb_tree::_pre_order(rb_avl_tree_base* node) const {
  if (node == nullptr) return;
  std::cout << "  " << node->_val << " | "
            << ((node->_color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
  _pre_order(node->_left);
  _pre_order(node->_right);
}

void rb_tree::_in_order(rb_avl_tree_base* node) const {
  if (node == nullptr) return;
  _in_order(node->_left);
  std::cout << "  " << node->_val << " | "
            << ((node->_color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
  _in_order(node->_right);
}

void rb_tree::_post_order(rb_avl_tree_base* node) const {
  if (node == nullptr) return;
  _post_order(node->_left);
  _post_order(node->_right);
  std::cout << "  " << node->_val << " | "
            << ((node->_color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
}

void rb_tree::bfs() {
  std::queue<rb_avl_tree_base*> stack;
  stack.push(_root);

  while (!stack.empty()) {
    if (stack.front()->_left != nullptr) stack.push(stack.front()->_left);
    if (stack.front()->_right != nullptr) stack.push(stack.front()->_right);
    std::cout << "  " << stack.front()->_val << " | "
              << (stack.front()->_color == _rb_tree_color::_red ? "red" : "black") << std::endl;
    stack.pop();
  }
}

void rb_tree::dfs() {
  std::stack<rb_avl_tree_base*> stack;
  stack.push(_root);

  while (!stack.empty()) {
    auto curr_node = stack.top();
    std::cout << "  " << stack.top()->_val << " | "
              << (stack.top()->_color == _rb_tree_color::_red ? "red" : "black") << std::endl;
    stack.pop();
    if (curr_node->_right != nullptr) stack.push(curr_node->_right);
    if (curr_node->_left != nullptr) stack.push(curr_node->_left);
  }
}

class avl_tree_base {
 public:
  // 平衡因子：左子树 减 右子树
  int val, balance_factor;
  avl_tree_base *parent, *left, *right;

  explicit avl_tree_base(int v = 0, int bf = 0, avl_tree_base* p = nullptr,
                         avl_tree_base* l = nullptr, avl_tree_base* r = nullptr)
      : val(v), balance_factor(bf), parent(p), left(l), right(r) {}
};

class tree_avl {
 public:
  tree_avl();
  explicit tree_avl(avl_tree_base* root);

  void insert_node(avl_tree_base* node);
  void insert_val(int val);
  void delete_node(avl_tree_base* node);
  void delete_val(int val);
  avl_tree_base* search_val(int val);

  int get_height();

  void pre_order();
  void in_order();
  void post_order();

 private:
  void _insert_fixup(avl_tree_base* node);
  void _delete_fixup(avl_tree_base* node);

  avl_tree_base* _rotate_ll(avl_tree_base* node);
  avl_tree_base* _rotate_rl(avl_tree_base* node);
  avl_tree_base* _rotate_rr(avl_tree_base* node);
  avl_tree_base* _rotate_lr(avl_tree_base* node);

  int _get_height(avl_tree_base* node);

  void _update_bf(avl_tree_base* node);

  void _pre_order(avl_tree_base* node);
  void _in_order(avl_tree_base* node);
  void _post_order(avl_tree_base* node);

 protected:
  avl_tree_base* _root;
};

tree_avl::tree_avl() : _root(nullptr) {}
tree_avl::tree_avl(avl_tree_base* root) : _root(root) {}

void tree_avl::insert_val(int val) { insert_node(new avl_tree_base(val)); }

void tree_avl::insert_node(avl_tree_base* node) {
  avl_tree_base *iter = _root, *p = nullptr;

  while (iter != nullptr) {
    p = iter;
    if (node->val < iter->val)
      iter = iter->left;
    else
      iter = iter->right;
  }
  node->parent = p;
  if (p == nullptr)
    _root = node;
  else {
    if (node->val < p->val)
      p->left = node;
    else
      p->right = node;
  }
  _insert_fixup(node);
}

void tree_avl::_insert_fixup(avl_tree_base* node) {
  avl_tree_base *p = node->parent, *iter = node;
  while (p) {
    // 更新平衡因子
    if (p->left == iter)
      ++p->balance_factor;
    else if (p->right == iter)
      --p->balance_factor;
    else
      break;

    // 检查平衡因子
    if (p->balance_factor == 0)
      break;
    else if (p->balance_factor == -1 || p->balance_factor == 1) {
      iter = p;
      p = p->parent;
    } else {
      if (p->balance_factor == 2) {
        if (iter->balance_factor == 1)
          _rotate_ll(p);
        else
          _rotate_lr(p);
      } else if (p->balance_factor == -2) {
        if (iter->balance_factor == -1)
          _rotate_rr(p);
        else
          _rotate_rl(p);
      } else
        break;
    }
  }
}

avl_tree_base* tree_avl::_rotate_rr(avl_tree_base* node) {
  avl_tree_base* ret = nullptr;
  if (node->parent == nullptr) {
    // 如果当前节点为根节点
    _root = node->right;
    ret = _root;
  } else {
    ret = node->right;
    if (node->parent->left == node)
      node->parent->left = ret;
    else
      node->parent->right = ret;
  }

  ret->parent = node->parent;
  node->parent = ret;
  node->right = ret->left;
  ret->left = node;

  // 更正平衡因子
  _update_bf(ret);

  return ret->right;
}

avl_tree_base* tree_avl::_rotate_rl(avl_tree_base* node) {
  avl_tree_base* ret = nullptr;

  if (node->parent == nullptr) {
    // 如果当前节点为根节点
    _root = node->right->left;
    ret = _root;
  } else {
    ret = node->right->left;
    if (node->parent->right == node)
      node->parent->right = ret;
    else
      node->parent->left = ret;
  }

  auto tmp = node->right;

  ret->parent = node->parent;

  node->parent = ret;
  node->right = ret->left;

  tmp->parent = ret;
  tmp->left = ret->right;

  ret->left = node;
  ret->right = tmp;

  // 更正平衡因子
  _update_bf(ret);

  return ret;
}

avl_tree_base* tree_avl::_rotate_ll(avl_tree_base* node) {
  avl_tree_base* ret = nullptr;
  if (node->parent == nullptr) {
    // 如果当前节点为根节点
    _root = node->left;
    ret = _root;
  } else {
    ret = node->left;
    if (node->parent->left == node)
      node->parent->left = ret;  // 当前节点为父节点的左节点
    else
      node->parent->right = ret;  // 当前节点为父节点的右节点
  }

  ret->parent = node->parent;
  node->left = ret->right;
  node->parent = ret;
  ret->right = node;

  // 更正平衡因子
  _update_bf(ret);

  return ret->left;
}

avl_tree_base* tree_avl::_rotate_lr(avl_tree_base* node) {
  avl_tree_base* ret = nullptr;

  if (node->parent == nullptr) {
    // 如果当前节点为根节点
    _root = node->left->right;
    ret = _root;
  } else {
    ret = node->left->right;
    if (node->parent->right == node)
      node->parent->right = ret;
    else
      node->parent->left = ret;
  }

  auto tmp = node->left;

  ret->parent = node->parent;

  node->parent = ret;
  node->left = ret->right;

  tmp->parent = ret;
  tmp->right = ret->left;

  ret->left = tmp;
  ret->right = node;

  // 更正平衡因子
  _update_bf(ret);

  return ret;
}

void tree_avl::_update_bf(avl_tree_base* node) {
  node->balance_factor = _get_height(node->left) - _get_height(node->right);
  node->left->balance_factor = _get_height(node->left->left) - _get_height(node->left->right);
  node->right->balance_factor = _get_height(node->right->left) - _get_height(node->right->right);
}

int tree_avl::get_height() { return _get_height(_root); }

int tree_avl::_get_height(avl_tree_base* node) {
  if (node == nullptr) return 0;
  return std::max(_get_height(node->left), _get_height(node->right)) + 1;
}

void tree_avl::delete_val(int val) { delete_node(search_val(val)); }

// todo:
void tree_avl::delete_node(avl_tree_base* node) {
  if (node == nullptr) return;
}

// todo:
void tree_avl::_delete_fixup(avl_tree_base* node) {}

avl_tree_base* tree_avl::search_val(int val) {
  avl_tree_base* iter = _root;
  while (iter != nullptr) {
    if (iter->val < val)
      iter = iter->right;
    else if (iter->val > val)
      iter = iter->left;
    else
      return iter;
  }
  return iter;
}

void tree_avl::pre_order() { _pre_order(_root); }
void tree_avl::in_order() { _in_order(_root); }
void tree_avl::post_order() { _post_order(_root); }

void tree_avl::_pre_order(avl_tree_base* node) {
  if (node == nullptr) return;
  std::cout << "  " << node->val << " | " << node->balance_factor << std::endl;
  _pre_order(node->left);
  _pre_order(node->right);
}

void tree_avl::_in_order(avl_tree_base* node) {
  if (node == nullptr) return;
  _in_order(node->left);
  std::cout << "  " << node->val << " | " << node->balance_factor << std::endl;
  _in_order(node->right);
}

void tree_avl::_post_order(avl_tree_base* node) {
  if (node == nullptr) return;
  _post_order(node->left);
  _post_order(node->right);
  std::cout << "  " << node->val << " | " << node->balance_factor << std::endl;
}

namespace test {
void test_rb_tree() {
  // 验证二叉树序列 前序和中序
  auto* new_tree = new rb_tree();
  for (int iter = 0; iter < 10; ++iter) new_tree->insert(iter);
  std::cout << "the pre order: " << std::endl;
  new_tree->pre_order();
  std::cout << "the in order: " << std::endl;
  new_tree->in_order();
  std::cout << "BFS order: " << std::endl;
  new_tree->bfs();
  std::cout << "DFS order: " << std::endl;
  new_tree->dfs();
}

void test_avl_tree() {
  std::vector<int> nums{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  auto* new_tree = new tree_avl();
  std::cout << "insert node: ";
  for (int num : nums) {
    new_tree->insert_val(num);
    std::cout << num << " ";
  }
  std::cout << std::endl;

  std::cout << "pre_order: " << std::endl;
  new_tree->pre_order();
  std::cout << "in order: " << std::endl;
  new_tree->in_order();

  std::cout << "avl_tree height: " << new_tree->get_height() << std::endl;
}
}  // namespace test
}  // namespace tiny_rb_tree

// todo: 增加view视图功能
#include <ranges>
namespace feature_ranges {
class point2d {
 public:
  int x, y, z;
};

namespace test {
void test_new_feature() {
  // 列表初始化
  point2d new_point{.x = 1, .y = 4, .z = 3};
  std::cout << new_point.x << " | " << new_point.y << " | " << new_point.z << std::endl;
  std::vector<int> ints{0, 1, 2, 3, 4, 5};
  for (int i : ints | std::views::filter([](int i) { return i % 2 == 0; }) |
                   std::views::transform([](int i) { return i * i; }))
    std::cout << i << std::endl;
}
}  // namespace test
}  // namespace feature_ranges

namespace test {
void test() {
  std::vector<std::vector<int>> a{{1, 2, 3}, {2, 3, 4}};
  for (auto& iter_row : a)
    for (auto& iter_col : iter_row) std::cout << iter_col << " ";
  std::cout << std::endl;
}
}  // namespace test

int main() {
  tiny_sort::test::test_sort();
  //  tiny_search::test::test_search();
  //  tiny_rb_tree::test::test_rb_tree();
  //  feature_ranges::test::test_new_feature();
  //  test::test();

  //  tiny_rb_tree::test::test_avl_tree();
  return 0;
}
