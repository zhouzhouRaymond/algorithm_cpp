#include <bits/stdc++.h>

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
 * 7. 随机快排
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

std::vector<int> heap_sort_swim(std::vector<int> num) {
  int nums_size = static_cast<int>(num.size()) - 1;
  for (int iter = nums_size / 2; iter > 0; --iter) _sink(num, iter, nums_size);

  for (int iter = 0; iter < nums_size;) {
    std::swap(num[iter], num[nums_size]);
    _swim(num, ++iter, nums_size);
  }
  return num;
}

std::vector<int> heap_sort_sink(std::vector<int> num) {
  int num_size = static_cast<int>(num.size()) - 1;
  for (int iter = num_size / 2; iter > 0; --iter) _sink(num, iter, num_size);

  for (int iter = num_size; iter > 0;) {
    std::swap(num[iter], num[0]);
    _sink(num, 0, --iter);
  }
  return num;
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

/* 随机排序与快排的差别主要在于
 * partition的位置不是直接选择第一个，而是随机选择
 * */
int _random_partition(std::vector<int>& nums, int low, int high) {
  std::swap(nums[low], nums[(rand() % (high - low)) + low]);
  return _partition(nums, low, high);
}

void _random_quick_sort(std::vector<int>& nums, int low, int high) {
  if (low >= (high - 1)) return;

  int partition = _random_partition(nums, low, high);
  _random_quick_sort(nums, low, partition);
  _random_quick_sort(nums, partition + 1, high);
}

std::vector<int> random_quick_sort(std::vector<int> nums) {
  _random_quick_sort(nums, 0, static_cast<int>(nums.size()));
  return nums;
}

namespace test {
void test_sort() {
  std::vector<int> num{1, 2, 3, 4, 5, 6, 7};
  //  std::cout << tiny_sort::middle_finder(num);
  //  auto tmp = tiny_sort::merge_high_to_low(num);
  //  auto tmp = tiny_sort::merge_low_to_high(num);
  //  std::cout << tiny_sort::inverse_pairs_important(num);

  //  auto tmp = tiny_sort::quick_sort(num);
  //  for (auto nums : tmp) std::cout << nums << " ";
  //  std::cout << select_index(num, 1);
  auto tmp = tiny_sort::random_quick_sort(num);
  for (auto item : tmp) std::cout << item << std::endl;
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

// todo: 红黑树删除节点
void rb_tree::delete_node(rb_avl_tree_base* node) {
  if (node == nullptr) return;
}

void rb_tree::delete_node(int val) { delete_node(search(val)); }

// todo: 红黑树删除节点修复
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
}  // namespace test
}  // namespace tiny_rb_tree

namespace tiny_avl_tree {
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
  void _delete_fixup(avl_tree_base* node, int left);

  avl_tree_base* _find_max(avl_tree_base* node);

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
    // 父节点挂载子树
    ret = node->right;
    if (node->parent->left == node)
      node->parent->left = ret;
    else
      node->parent->right = ret;
  }

  ret->parent = node->parent;
  node->parent = ret;
  node->right = ret->left;
  if (ret->left != nullptr) ret->left->parent = node;
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
  node->parent = ret;
  node->left = ret->right;
  if (ret->right != nullptr) ret->right->parent = node;
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

/* 删除节点三种情况：
 * 1. 叶子节点，直接删除
 * 2. 只有一颗子树，提升子树，并更新平衡因子
 * 3. 有两颗子树，选择左子树中最大的一个替换当前节点，并更新平衡因子
 * */
void tree_avl::delete_node(avl_tree_base* node) {
  if (node == nullptr) return;
  // 如果存在父节点，将父节点的孩子节点置为空，并指示左右 -1 左 1 右 0 不存在
  int left = 0;
  auto node_parent = node->parent;
  if (node_parent != nullptr) {
    if (node_parent->left == node) {
      node_parent->left = nullptr;
      left = -1;
    } else {
      node_parent->right = nullptr;
      left = 1;
    }
  }
  // 后两种情况
  if (node->left == nullptr && node->right != nullptr) {
    // 子树指向父节点 且 父节点链接子树
    node->right->parent = node_parent;
    if (left != 0 && left == 1)
      node_parent->right = node->right;
    else if (left != 0 && left == -1)
      node_parent->left = node->right;

    // 子树节点挂载
    node->right->parent = node_parent;
    // 当前节点变更为叶子节点
    node->right = nullptr;
  } else if (node->left != nullptr && node->right == nullptr) {
    node->left->parent = node_parent;
    if (left != 0 && left == 1)
      node_parent->right = node->left;
    else if (left != 0 && left == -1)
      node_parent->left = node->left;
    node->left->parent = node_parent;
    node->left = nullptr;
  } else if (node->left != nullptr && node->right != nullptr) {
    // 找到左子树中最大的一个
    auto node_max = _find_max(node->left);
    if (left == 0)
      // 置换根节点
      _root = node_max;
    // 替换当前节点
    node_max->left = node->left;
    node_max->right = node->right;
    node_max->balance_factor = node->balance_factor;

    node->left->parent = node_max;
    node->right->parent = node_max;

    node->left = nullptr;
    node->right = nullptr;

    node->parent = node_max->parent;

    if (node_max->parent->left == node_max)
      node_max->parent->left = node;
    else
      node_max->parent->right = node;

    node_max->parent = node_parent;

    // 父节点挂载
    if (left != 0 && left == 1)
      node_parent->right = node_max;
    else if (left != 0 && left == -1)
      node_parent->left = node_max;
  }
  // 要删除的节点变为叶子节点 直接删除
  _delete_fixup(node, left);
}

// left指示当前节点是父节点的左右孩子 -1 左 1 右 0 不存在父节点
void tree_avl::_delete_fixup(avl_tree_base* node, int left) {
  // 删除当前节点，并向上修复平衡因子
  // 卸载父节点的指引
  if (node != nullptr && left != 0 && node->parent->left == node)
    node->parent->left = nullptr;
  else if (node != nullptr && left != 0 && node->parent->right == node)
    node->parent->right = nullptr;
  else if (left == 0)
    // 当前节点为根节点
    _root = nullptr;
  auto iter_node = node->parent;
  delete node;

  // 向上修改平衡因子
  while (iter_node != nullptr) {
    // 修改当前节点的平衡因子
    iter_node->balance_factor = _get_height(iter_node->left) - _get_height(iter_node->right);
    /* 三种情况
     * 1. 当前节点的平衡因子为 -1 或 1 表明当前节点的高度并没有变化
     * 2. 当前节点的平衡银子为 2 或 -2 节点已经失衡 需要调整
     *    2.1 平衡因子为 2 左孩子节点的平衡因子为 0 或 -1 则 LR旋转 左孩子节点为 1 则LL旋转
     *    2.2 平衡因子为 -2 右孩子节点的平衡因子为 0 或 1 则 RL旋转，右孩子为 -1 则RR旋转　
     * 3. 当前节点的平衡因子为 0 表明 子树高度减少，应当向上修改平衡因子
     * */
    if (iter_node->balance_factor == -1 || iter_node->balance_factor == 1)
      break;
    else if (iter_node->balance_factor == 2) {
      if (iter_node->left->balance_factor == 1)
        _rotate_ll(iter_node);
      else
        _rotate_lr(iter_node);
    } else if (iter_node->balance_factor == -2) {
      if (iter_node->balance_factor == -1)
        _rotate_rr(iter_node);
      else
        _rotate_rl(iter_node);
    } else if (iter_node->balance_factor == 0) {
      iter_node = iter_node->parent;
    }
  }
}

/* 返回当前树中的最大值节点 */
avl_tree_base* tree_avl::_find_max(avl_tree_base* node) {
  while (node->right != nullptr) node = node->right;
  return node;
}

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
void test_avl_tree() {
  std::vector<int> nums{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  auto* new_tree = new tree_avl();
  std::cout << "insert node: ";
  for (int num : nums) {
    new_tree->insert_val(num);
    std::cout << num << " ";
  }
  std::cout << std::endl;

  for (auto iter = nums.rbegin(), iter_end = nums.rend() - 3; iter != iter_end; ++iter)
    new_tree->delete_val(*iter);

  std::cout << "pre_order: " << std::endl;
  new_tree->pre_order();
  std::cout << "in order: " << std::endl;
  new_tree->in_order();

  std::cout << "avl_tree height: " << new_tree->get_height() << std::endl;
}
}  // namespace test
}  // namespace tiny_avl_tree

namespace test_new_feature {
#include <ranges>
class point2d {
 public:
  int x, y, z;
};
// todo: 增加view视图特性

void test() {
  // 列表初始化
  point2d new_point{.x = 1, .y = 4, .z = 3};
  std::cout << new_point.x << " | " << new_point.y << " | " << new_point.z << std::endl;
  std::vector<int> ints{0, 1, 2, 3, 4, 5};
  for (int i : ints | std::views::filter([](int i) { return i % 2 == 0; }) |
                   std::views::transform([](int i) { return i * i; }))
    std::cout << i << std::endl;
}
}  // namespace test_new_feature

namespace more_effective_cxx {
void test_01() {
  // reference 总是指向一个非空的初值，所以在初始化时需要赋值
  // pointer 指向一个空位置，并且可以改变指向的对象
}

void test_02() {
  // static_cast 转型操作
  // const_cast 去除const或添加const
  // dynamic_cast 指向基类的派生类 转型为指针 转型失败返回null 转型为引用 转型失败 抛出一个异常
  // reinterpret_cast 强制转换函数指针
  // typedef void (*funcptr)(); funcptr funcptrarry[10]; int dosomething();
  // funcptrarry[0] = reinterpret_cast<funcptr>(&dosomething);
}

void test_03() {
  // 1. 避免以具体类继承具体类
  // 2. 避免以多态的方式处理数组，指针在移动时无法确定的知道下一个位置
}

void test_04() {
  // 如非必要，不提供default constructors
}

void test_05() {
  // 只有含有单一自变量类型的函数可以成员隐式类型转换函数
  // 使用explicit修饰构造函数 使此函数避免隐式类型转换
}

void test_06() {
  // 前置 和 后置 ++/--
  // T& operator++(); 前置  累加然后取出 返回的是当前值的引用
  // const T operator++(int); 后置  取出然后累加 返回的是旧值
  // T& operator+=(int); +=操作符
  // 两者的返回值不同

  // 后置累加 以 前置累加为基础
}

void test_07() {
  // 不可以重载
  // && || , . .* :: ?: new delete sizeof typeid static/dynamic/const/reinterpret_cast

  // 可以重载
  // operator new/new[] operate delete/new[] + - * / % ^ & | ~ ! = < >
  // +/-/*///%/^/&/|/<</>>/=/!/</>= ++ -- ->* -> () []
}

void test_08() {
  // new/delete/new[]/delete[] operator 和 operator new/delete/new[]/delete[] 的区别
  // 前者为系统内建的语言，实现两个功能 1. 分配原始内存，2. 创建对象
  // 后者为重载类型,只能实现开辟原始内存的功能

  // placement new/delete/new[]/delete[] 在为初始化的原始内存上创建对象
  // new (placement of buffer) T(sizeof(T));
}

void test_09() {
  // 把资源对象封装在对象内,则可以极大程度上避免资源泄漏
  // 尽可能多的使用智能指针
}

void test_10() {
  // 在构造函数内消除资源泄露的困扰
  // 以资源管理类的形式管理资源
}

void test_11() {
  // 应当在 析构函数 中处理所有异常
}

void test_12() {
  // 抛出异常总是会发生拷贝,处理异常时,其实是处理抛出异常的一个副本
  // 且在处理异常时,总是只考虑异常的静态类型,而忽略异常的动态类型
  // throw; throw w; 前者抛出的是当前类型, 后者抛出的是当前类型的一个拷贝

  // 应当避免以 pass-by-point 传递异常,因为抛出的异常会在抛出之后析构掉

  // 抛出的异常不存在隐式类型转换
  // 只存在两种类型转换:
  // 1. 基类和子类,抛出一个子类的异常可以被基类捕获
  // 2. 从有型指针 转换为 五型指针

  // 在处理异常时 总时以最先吻合策略处理 而在处理函数时,总是以最佳吻合处理
}

void test_13() {
  // 以 pass-by-reference 捕获异常参数
}

void test_14() {
  // 不为template提供异常列表
  // 如果A函数内调用函数B,而B函数无异常列表,那么函数A也不应该设置异常列表
  // 处理系统可能抛出的异常,最常见的就是bad_alloc,内存分配失败
}

void test_15() {
  // 处理异常需要时间和空间成本
}

void test_16() {
  // 80 - 20 法则
  // 使用性能分析器 尽可能多的使用不同的数据分析程序性能
}

void test_17() {
  // 缓式评估
  // 引用计数 当程序确切的需要修改自身数据时，才私有化数据，否则采用引用计数的方式处理数据
  // 区分读和写
  // 表达式缓式评估 只在需要真正使用表达式值的时候才计算使用的那部分
}

void test_18() {
  // 超急式评估 考虑到程序需要使用某值时，分摊计算此值的成本，比如 数据的平均值，最大值，最小值
  // 缓存技术
  // 预先取出 比如 在动态数组中每次分配比需求大的空间
}

void test_19() {
  // 临时对象：1. 隐式类型转换 2. 函数返回对象
  // const reference 会产生临时对象 non-const reference 则不会产生历史对象
  // 针对函数返回对象所产生的临时对象，可采用返回值优化
}

void test_20() {
  // 函数必须要返回对象时
  // inline const T operator *(const T& lhs, const T& rhs){ return T(lhs * rhs); }
  // 编译器内部优化了函数返回值的构造和析构成本，使此临时对象的在接收返回值的对象内部构造
}

void test_21() {
  // 使用重载技术，为函数指明参数类型，避免隐式类型转换
  // T a = b * c; b, c is a T
  // T a = b * c; b is a T, c is not a T
  // 不指明参数类型时，c会进行隐式类型转换
  // 且在重载操作符时，必须有一个参数是用户自定义的类型
}

void test_22() {
  // 通过操作符的符合版本实现独身版本
  // + - += -+
}

void test_23() {
  // 使用不同的程序库
}

void test_24() {
  // 无法将虚函数声明为inline，因为inline函数，在编译器就知道了函数调用，而virtual函数只有在运行期才知道
  // 在单一继承中，对象的布局只是增加了一个vptr成本
  // 在多继承中，因为virtual base基类的问题，需要要中间类指定 pointer to virtual base class
  // 并把base class的成员放到最下方

  // 运行期 取得对象的类型信息 只需要在虚函数表的最上面添加 对象的类型信息指针即可
}

void test_25() {
  // 虚化 构造函数 和 non-member函数

  // 复制构造函数：
  // 例子：有一个链表指向基类的派生类对象，当需要复制此链表时，需要虚化复制构造函数
  // 有一个可行的例子就是：
  // virtual base * clone () const = 0;
  // virtual drived * clone() const{ return new drived(*this); }

  // cout 函数
  // base class: virtual ostream& print(ostream& s) const = 0;
  // drived class: virtual ostream& print(ostream& s) const{ }
  // inline ostream& operator<<(ostream& s, const base& c) { return c.print(s); }
}

void test_26() {
  // 限制对象所能产出的对象数量
  // 单例模式 和 限制模式
  // 单例模式 在函数内使用 static对象,使其只有在使用时才会初始化 并且只初始化一次
  // 限制模式 创建一个计数的基类 现在对象生成的最大个数
}

void test_27() {
  // 限制对象必须产生于heap内,private/protect ~dtor
  // 判断对象是否在heap中,operator new 操作符中记录每次new出的地址空间起始位置,
  // 使用dynamic_cast将类型转换为void* 再与记录的地址比较,则可以判断出是否包含在heap中

  // 禁止对象产生于heap中
  // 私有化 operator new/delete
  // 对于含有自己operator new函数的派生类,基类的私有性不具影响
}

void test_28() {
  // 智能指针内部不可以提供对原始指针的隐式类型转换

  // 当提供隐式类型转换时，需要特别注意类的多继承

  // non-const to const的类型转换
}

void test_29() {
  //
}

void test_30() {
  //
}

void test_31() {
  //
}

void test_32() {
  //
}

void test_33() {
  //
}

void test_34() {
  //
}

void test_35() {
  //
}

}  // namespace more_effective_cxx

int main() {
  //  tiny_avl_tree::test::test_avl_tree();
  tiny_sort::test::test_sort();
  return 0;
}
