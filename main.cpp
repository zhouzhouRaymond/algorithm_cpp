#include <algorithm>
#include <iostream>
#include <queue>
#include <ranges>
#include <stack>
#include <string_view>
#include <vector>

#include "dbg-macro/dbg.h"

namespace tiny_sort {
std::vector<int> select_sort(std::vector<int> num) {
  int num_size = static_cast<int>(num.size());
  // min is current changing position
  for (int min = 0; min < num_size; ++min) {
    int iter_min = min;
    // find the min_index from the unsorted part
    for (int iter = min; iter < num_size; ++iter)
      iter_min = num[iter_min] > num[iter] ? iter : iter_min;
    // change
    std::swap(num[min], num[iter_min]);
  }
  return num;
}

std::vector<int> insert_sort(std::vector<int> num) {
  int num_size = static_cast<int>(num.size());
  for (int iter = 1; iter < num_size; ++iter) {
    int iter_min = 0;
    // find the insert position
    while (iter_min <= iter && num[iter_min] <= num[iter]) ++iter_min;
    // move for current position
    for (; iter_min < iter; ++iter_min) std::swap(num[iter_min], num[iter]);
  }
  return num;
}

std::vector<int> shell_sort(std::vector<int> num) {
  int h = 1, num_size = static_cast<int>(num.size());
  while (h < num_size / 3) h = 3 * h + 1;
  for (; h >= 1; h /= 3)
    // h-sorted
    for (int iter = h; iter < num_size; ++iter)
      for (int iter_min = iter; iter_min >= h && num[iter_min] < num[iter_min - h]; iter_min -= h)
        std::swap(num[iter_min], num[iter_min - h]);

  return num;
}

// the index is [low, high)
void _merge(std::vector<int>& num, int low, int middle, int high) {
  if (num[middle] <= num[middle + 1]) return;
  std::vector<int> ret;
  int start = low, ret_middle = middle++;
  for (; low <= ret_middle && middle <= high;) {
    if (num[low] <= num[middle])
      ret.push_back(num[low++]);
    else
      ret.push_back(num[middle++]);
  }
  for (; low <= ret_middle; ++low) ret.push_back(num[low]);
  for (; middle <= high; ++middle) ret.push_back(num[middle]);

  (void)std::for_each(ret.begin(), ret.end(), [&start, &num](int& val) { num[start++] = val; });
}

// the index is [low, high]
void merge_high_to_low(std::vector<int>& num, int low, int high) {
  if (low == high) return;
  int middle = low + (high - low) / 2;
  merge_high_to_low(num, low, middle);
  merge_high_to_low(num, middle + 1, high);
  // merge into num
  _merge(num, low, middle, high);
}

void merge_low_to_high(std::vector<int>& num) {
  int N = num.size();
  for (int sz = 2; sz <= N; sz *= 2)
    for (int iter = 0; iter + sz - 1 < N; iter += sz)
      _merge(num, iter, iter + sz / 2 - 1, iter + sz - 1);

  num = insert_sort(num);
}

int _inverse_pairs(std::vector<int>& num, int low, int high) {
  if (low == high) return 0;
  int ret = 0;
  int middle = low + (high - low) / 2;

  ret += _inverse_pairs(num, low, middle);
  ret += _inverse_pairs(num, middle + 1, high);

  for (int left = middle, right = high; left >= low && high > middle;) {
    if (num[left] > num[right]) {
      ret += right - middle;
      --left;
    } else if (num[left] < num[right])
      --right;
    else
      break;
  }
  _merge(num, low, middle, high);
  return ret;
}

int inverse_pairs(std::vector<int> num) {
  return _inverse_pairs(num, 0, static_cast<int>(num.size()) - 1);
}

int _reverse_pairs_important(std::vector<int>& num, int low, int high) {
  if (low >= high) return 0;
  int ret = 0, middle = (high - low) / 2 + low;
  ret += _reverse_pairs_important(num, low, middle);
  ret += _reverse_pairs_important(num, middle + 1, high);

  // 统计重要逆序对
  for (int iter_low = low, iter_high = middle + 1; iter_low <= middle; ++iter_low) {
    while (iter_high <= high && num[iter_low] > num[iter_high] * 2LL) ++iter_high;
    ret += iter_high - (middle + 1);
  }
  _merge(num, low, middle, high);
  return ret;
}

int inverse_pairs_important(std::vector<int> num) {
  return _reverse_pairs_important(num, 0, static_cast<int>(num.size()) - 1);
}

int _partition(std::vector<int>& num, int low, int high) {
  int iter_low = low, iter_high = high;
  while (iter_low != iter_high) {
    while (num[iter_low] <= num[low] && iter_low < high) ++iter_low;
    while (num[iter_high] >= num[low] && iter_high > low) --iter_high;
    if (iter_low > iter_high) break;
    std::swap(num[iter_low], num[iter_high]);
  }
  std::swap(num[low], num[iter_high]);
  return iter_high;
}

void quick_sort(std::vector<int>& num, int low, int high) {
  if (low >= high) return;

  int partition = _partition(num, low, high);
  quick_sort(num, low, partition - 1);
  quick_sort(num, partition + 1, high);
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

int midian_finder(const std::vector<int>& num) {
  int ret = 0;
  std::priority_queue<int> max_heap;
  std::priority_queue<int, std::vector<int>, std::greater<>> min_heap;

  for (int item : num) {
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

int select_index(std::vector<int> num, int k) {
  k = k - 1;
  int low = 0, high = static_cast<int>(num.size()) - 1;
  while (low < high) {
    int iter = _partition(num, low, high);
    if (iter == k)
      return num[k];
    else if (iter > k)
      high = iter - 1;
    else
      low = iter + 1;
  }
  return num[k];
}

namespace test {
void test_sort() {
  std::vector<int> num{9, 8, 7, 6, 5, 4, 3, 2, 1};

  std::cout << "the inverse pairs: " << tiny_sort::inverse_pairs(num) << std::endl;
  std::cout << "the inverse pairs: " << tiny_sort::inverse_pairs_important(num) << std::endl;

  std::vector<int> ret_select = tiny_sort::select_sort(num);
  std::vector<int> ret_insert = tiny_sort::insert_sort(num);
  std::vector<int> ret_shell = tiny_sort::shell_sort(num);
  std::vector<int> vec_merge_lh = num, vec_merge_hl = num, vec_quick = num, vec_heap_sink = num,
                   vec_heap_swim = num;
  tiny_sort::merge_high_to_low(vec_merge_hl, 0, static_cast<int>(num.size()) - 1);
  tiny_sort::merge_low_to_high(vec_merge_lh);
  tiny_sort::quick_sort(vec_quick, 0, static_cast<int>(vec_quick.size()) - 1);
  tiny_sort::heap_sort_sink(vec_heap_sink);

  std::cout << "   select sort: " << std::is_sorted(ret_select.begin(), ret_select.end())
            << std::endl;
  std::cout << "   insert sort: " << std::is_sorted(ret_insert.begin(), ret_insert.end())
            << std::endl;
  std::cout << "   shell sort: " << std::is_sorted(ret_shell.begin(), ret_shell.end()) << std::endl;
  std::cout << "   merge_hl sort: " << std::is_sorted(vec_merge_hl.begin(), vec_merge_hl.end())
            << std::endl;
  std::cout << "   merge_lh sort: " << std::is_sorted(vec_merge_lh.begin(), vec_merge_lh.end())
            << std::endl;
  std::cout << "   quick sort: " << std::is_sorted(vec_quick.begin(), vec_quick.end()) << std::endl;
  std::cout << "   heap sort_sink: " << std::is_sorted(vec_heap_sink.begin(), vec_heap_sink.end())
            << std::endl;
  std::cout << "   midian number: " << tiny_sort::midian_finder(num) << std::endl;
  std::cout << "   select the 6th number: " << tiny_sort::select_index(num, 6) << std::endl;
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
  int val;
  rb_avl_tree_base* parent;
  rb_avl_tree_base* left;
  rb_avl_tree_base* right;
  _rb_tree_color color;

  explicit rb_avl_tree_base(int value = 0, rb_avl_tree_base* p = nullptr,
                            rb_avl_tree_base* l = nullptr, rb_avl_tree_base* r = nullptr,
                            _rb_tree_color c = _red)
      : val(value), parent(p), left(l), right(r), color(c) {}
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

  void pre_order_for();
  void in_order_for();
  void post_order_for();

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
  auto tmp = node->right;
  node->right = tmp->left;
  if (tmp->left != nullptr) tmp->left->parent = node;
  tmp->parent = node->parent;
  if (node->parent == nullptr)
    _root = tmp;
  else if (node == node->parent->left)
    node->parent->left = tmp;
  else
    node->parent->right = tmp;
  tmp->left = node;
  node->parent = tmp;
}

void rb_tree::_right_rotate(rb_avl_tree_base* node) {
  auto tmp = node->left;
  node->left = tmp->right;
  if (tmp->right != nullptr) tmp->right->parent = node;
  tmp->parent = node->parent;
  if (node->parent == nullptr)
    _root = tmp;
  else if (node == node->parent->left)
    node->parent->left = tmp;
  else
    node->parent->right = tmp;
  tmp->right = node;
  node->parent = tmp;
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
    if (node->val < iter->val)
      iter = iter->left;
    else
      iter = iter->right;
  }
  node->parent = p;
  if (p != nullptr) {
    if (node->val < p->val)
      p->left = node;
    else
      p->right = node;
  } else
    _root = node;
  node->color = _rb_tree_color::_red;
  _insert_fixup(node);
}

void rb_tree::_insert_fixup(rb_avl_tree_base* node) {
  while (node->parent != nullptr && node->parent->color == _rb_tree_color::_red) {
    if (node->parent == node->parent->parent->left) {
      auto tmp_right = node->parent->parent->right;
      if (tmp_right != nullptr && tmp_right->color == _rb_tree_color::_red) {
        node->parent->color = _rb_tree_color::_black;
        tmp_right->color = _rb_tree_color::_black;
        node->parent->parent->color = _rb_tree_color::_red;
        node = node->parent->parent;
      } else if (node == node->parent->right) {
        node = node->parent;
        _left_rotate(node);
      } else {
        node->parent->color = _rb_tree_color::_black;
        node->parent->parent->color = _rb_tree_color::_red;
        _right_rotate(node->parent->parent);
      }
    } else {  // the right part
      auto tmp_left = node->parent->parent->left;
      if (tmp_left != nullptr && tmp_left->color == _rb_tree_color::_red) {
        node->parent->color = _rb_tree_color::_black;
        tmp_left->color = _rb_tree_color::_black;
        node->parent->parent->color = _rb_tree_color::_red;
        node = node->parent->parent;
      } else if (node == node->parent->left) {
        node = node->parent;
        _right_rotate(node);
      } else {
        node->parent->color = _rb_tree_color::_black;
        node->parent->parent->color = _rb_tree_color::_red;
        _left_rotate(node->parent->parent);
      }
    }
  }
  _root->color = _rb_tree_color::_black;
}

// todo:
void rb_tree::delete_node(rb_avl_tree_base* node) {
  if (node == nullptr) return;
}

// todo:
void rb_tree::delete_node(int val) { delete_node(search(val)); }

// todo:
void rb_tree::_delete_fixup(rb_avl_tree_base* node) {}

void rb_tree::_delete_transplant(rb_avl_tree_base* node, rb_avl_tree_base* new_node) {
  if (node->parent == nullptr)
    _root = new_node;
  else if (node == node->parent->left)
    node->parent->left = new_node;
  else
    node->parent->right = new_node;
  new_node->parent = node->parent;
}

rb_avl_tree_base* rb_tree::search(int value) {
  rb_avl_tree_base* iter = _root;
  while (iter != nullptr) {
    if (iter->val == value)
      return iter;
    else if (iter->val > value)
      iter = iter->left;
    else
      iter = iter->right;
  }
  return nullptr;
}

/* 1. 访问当前节点，并将节点入栈
 * 2. 当前节点左孩子为空，将当前节点替换为栈顶节点的右孩子节点，执行操作1
 * 3. 当前节点左孩子不为空，替换当前节点为左孩子节点，执行操作1
 * */
void rb_tree::pre_order_for() {
  auto iter_node = _root;
  std::stack<rb_avl_tree_base*> nodes;
  while (iter_node != nullptr || nodes.empty() == false) {
    while (iter_node != nullptr) {
      std::cout << iter_node->val << " | "
                << ((iter_node->color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
      nodes.push(iter_node);
      iter_node = iter_node->left;
    }
    if (nodes.empty() == false) {
      iter_node = nodes.top();
      nodes.pop();
      iter_node = iter_node->right;
    }
  }
}

/* 1. 将当前节点入栈
 * 2. 当前节点左孩子为空，输出栈顶节点，将当前节点替换为栈顶结点的右孩子节点，执行操作1
 * 3. 当前节点左孩子不为空，将当前节点替换为左孩子节点，执行操作1
 * */
void rb_tree::in_order_for() {
  auto iter_node = _root;
  std::stack<rb_avl_tree_base*> nodes;
  while (iter_node != nullptr || nodes.empty() == false) {
    while (iter_node != nullptr) {
      nodes.push(iter_node);
      iter_node = iter_node->left;
    }
    if (nodes.empty() == false) {
      iter_node = nodes.top();
      std::cout << iter_node->val << " | "
                << ((iter_node->color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
      nodes.pop();
      iter_node = iter_node->right;
    }
  }
}

/* 记录上一次输出的节点。
 * 当前节点在左右孩子访问之后再访问。
 * 当前节点的左右孩子节点为空 或 当前的节点的左右孩子节点已经被访问过 则直接输出
 * 否则 将当前节点的右左孩子依次入栈
 * */
void rb_tree::post_order_for() {
  rb_avl_tree_base *iter_node = nullptr, *pre = nullptr;
  std::stack<rb_avl_tree_base*> nodes;
  nodes.push(_root);
  while (nodes.empty() == false) {
    iter_node = nodes.top();
    if ((iter_node->left == nullptr && iter_node->right == nullptr) ||
        (pre != nullptr) && (iter_node->left == pre || iter_node->right == pre)) {
      // 当前节点不存在左孩子或右孩子 或 当前节点的孩子节点已经被访问过 则直接输出
      std::cout << iter_node->val << " | "
                << ((iter_node->color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
      nodes.pop();
      pre = iter_node;
    } else {
      if (iter_node->right != nullptr) nodes.push(iter_node->right);
      if (iter_node->left != nullptr) nodes.push(iter_node->left);
    }
  }
}

// 输出格式 value | color
void rb_tree::pre_order() { _pre_order(_root); }

void rb_tree::in_order() { _in_order(_root); }

void rb_tree::post_order() { _post_order(_root); }

void rb_tree::_pre_order(rb_avl_tree_base* node) const {
  if (node == nullptr) return;
  std::cout << "  " << node->val << " | "
            << ((node->color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
  _pre_order(node->left);
  _pre_order(node->right);
}

void rb_tree::_in_order(rb_avl_tree_base* node) const {
  if (node == nullptr) return;
  _in_order(node->left);
  std::cout << "  " << node->val << " | "
            << ((node->color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
  _in_order(node->right);
}

void rb_tree::_post_order(rb_avl_tree_base* node) const {
  if (node == nullptr) return;
  _post_order(node->left);
  _post_order(node->right);
  std::cout << "  " << node->val << " | "
            << ((node->color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
}

void rb_tree::bfs() {
  std::queue<rb_avl_tree_base*> stack;
  stack.push(_root);

  while (!stack.empty()) {
    if (stack.front()->left != nullptr) stack.push(stack.front()->left);
    if (stack.front()->right != nullptr) stack.push(stack.front()->right);
    std::cout << "  " << stack.front()->val << " | "
              << (stack.front()->color == _rb_tree_color::_red ? "red" : "black") << std::endl;
    stack.pop();
  }
}

void rb_tree::dfs() {
  std::stack<rb_avl_tree_base*> stack;
  stack.push(_root);

  while (!stack.empty()) {
    auto curr_node = stack.top();
    std::cout << "  " << stack.top()->val << " | "
              << (stack.top()->color == _rb_tree_color::_red ? "red" : "black") << std::endl;
    stack.pop();
    if (curr_node->right != nullptr) stack.push(curr_node->right);
    if (curr_node->left != nullptr) stack.push(curr_node->left);
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
  std::cout << "the post order: " << std::endl;
  new_tree->post_order();
  std::cout << "BFS order: " << std::endl;
  new_tree->bfs();
  std::cout << "DFS order: " << std::endl;
  new_tree->dfs();

  std::cout << "the pre order for loop: " << std::endl;
  new_tree->pre_order_for();
  std::cout << "the in order for loop: " << std::endl;
  new_tree->in_order_for();
  std::cout << "the post order for loop: " << std::endl;
  new_tree->post_order_for();
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
  //  tiny_sort::test::test_sort();
  //  tiny_search::test::test_search();
  tiny_rb_tree::test::test_rb_tree();
  //  feature_ranges::test::test_new_feature();
  //  test::test();

  //  tiny_rb_tree::test::test_avl_tree();
  return 0;
}
