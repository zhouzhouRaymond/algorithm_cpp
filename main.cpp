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
 */
namespace tiny_sort {
/* 选择排序
 * 从小到大依次排序
 * 稳定排序
 */
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
 */
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
 * 非稳定排序
 */
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
 * 非稳定排序
 */
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
 */
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
 */
int _random_partition(std::vector<int>& nums, int low, int high) {
  std::swap(nums[low], nums[(std::rand() % (high - low)) + low]);
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

class rb_tree_base {
 public:
  int val;
  rb_tree_base* parent;
  rb_tree_base* left;
  rb_tree_base* right;
  _rb_tree_color color;

  explicit rb_tree_base(int value = 0, rb_tree_base* p = nullptr, rb_tree_base* l = nullptr,
                        rb_tree_base* r = nullptr, _rb_tree_color c = _red)
      : val(value), parent(p), left(l), right(r), color(c) {}
};

/* 红黑树的性质：
 * 1. 节点是红色或黑色。
 * 2. 根是黑色。
 * 3. 所有叶子都是黑色（叶子是NIL节点）。
 * 4. 每个红色节点必须有两个黑色的子节点。（从每个叶子到根的所有路径上不能有两个连续的红色节点。）
 * 5. 从任一节点到其每个叶子的所有简单路径都包含相同数目的黑色节点。
 *
 * 最重要的是 性质四 和 性质五
 */
class rb_tree {
 public:
  rb_tree();
  explicit rb_tree(rb_tree_base* root);

  void insert(rb_tree_base* node);
  void insert(int val);
  void delete_node(rb_tree_base* node);
  void delete_node(int val);

  rb_tree_base* search(int value);

  void pre_order();
  void in_order();
  void post_order();

  void pre_order_for();
  void in_order_for();
  void post_order_for();

  void bfs();
  void dfs();

 private:
  void _left_rotate(rb_tree_base* node);
  void _right_rotate(rb_tree_base* node);

  void _insert_fixup(rb_tree_base* node);
  void _delete_fixup(rb_tree_base* node, int left, _rb_tree_color orgin_color);

  rb_tree_base* _find_max(rb_tree_base* node);
  rb_tree_base* _find_min(rb_tree_base* node);

  void _pre_order(rb_tree_base* node) const;
  void _in_order(rb_tree_base* node) const;
  void _post_order(rb_tree_base* node) const;

 public:
 private:
  rb_tree_base* _root;
};

rb_tree::rb_tree() : _root(nullptr) {}

rb_tree::rb_tree(rb_tree_base* root) : _root(root) {}

void rb_tree::insert(int val) {
  insert(new rb_tree_base(val, nullptr, nullptr, nullptr, _rb_tree_color::_red));
}

/* 插入节点
 * 1. 寻找插入位置
 * 2. 默认节点颜色为红色
 * 3. 修复新插入的节点
 */
void rb_tree::insert(rb_tree_base* node) {
  rb_tree_base *iter = _root, *p = nullptr;

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

/* 插入节点修复
 * 1. 插入节点为根，颜色改为黑色
 * 2. 插入节点的父节点是黑色的，则插入完成
 * 3. 插入节点的父节点是红色的，需要修复，且继续向上调整，直到到达根 或 满足性质
 * 4. 如果根修改之后为红色，需要更改根节点颜色
 */
void rb_tree::_insert_fixup(rb_tree_base* node) {
  while (node->parent != nullptr && node->parent->color == _rb_tree_color::_red) {
    // 当前节点为红色，且其父节点也为红色
    if (node->parent == node->parent->parent->left) {
      // 当前父亲节点位于祖父节点的左子树上
      auto tmp_right = node->parent->parent->right;
      if (tmp_right != nullptr && tmp_right->color == _rb_tree_color::_red) {
        // 叔父节点同样为红色 重新着色 当前节点迭代为祖父节点
        /*      黑             红
         *     / \            / \
         *    红  红 =>      黑  黑
         *   /              /
         *  红             红
         */
        node->parent->color = _rb_tree_color::_black;
        tmp_right->color = _rb_tree_color::_black;
        node->parent->parent->color = _rb_tree_color::_red;
        node = node->parent->parent;
      } else if (node == node->parent->right) {
        // 叔父节点颜色 不知道 或 黑色
        // 当前节点是父节点的右孩子 且 父节点是 祖父节点的左孩子
        /*      黑            黑
         *     /             /
         *    红    =>      红
         *     \           /
         *      红        红  <= node
         */
        node = node->parent;
        _left_rotate(node);
      } else {
        // 叔父节点颜色 不知道 或 黑色
        // 当前节点是父节点的右孩子 且 父节点是 祖父节点的左孩子
        /*      黑             红                   黑
         *     / \            / \                 / \
         *    红  黑   =>    黑   黑   =>  node=> 红  红
         *   /              /                        \
         *  红             红  <= node                黑
         */
        node->parent->color = _rb_tree_color::_black;
        node->parent->parent->color = _rb_tree_color::_red;
        _right_rotate(node->parent->parent);
      }
    } else {
      // 当前父亲节点位于祖父节点的右子树上
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
  // 保证根节点的颜色为黑
  _root->color = _rb_tree_color::_black;
}

/* 左旋 相当于 RR旋转 */
void rb_tree::_left_rotate(rb_tree_base* node) {
  rb_tree_base* tmp = node->right;
  if (node->parent == nullptr) {
    // 当前节点为根节点
    _root = tmp;
  } else {
    // 挂载父节点
    if (node->parent->left == node)
      node->parent->left = tmp;
    else
      node->parent->right = tmp;
  }

  tmp->parent = node->parent;
  node->parent = tmp;
  node->right = tmp->left;
  if (tmp->left != nullptr) tmp->left->parent = node;
  tmp->left = node;
}

/* 右旋 相当于 LL旋转 */
void rb_tree::_right_rotate(rb_tree_base* node) {
  rb_tree_base* tmp = node->left;
  if (node->parent == nullptr) {
    _root = tmp;
  } else {
    if (node->parent->left == node)
      node->parent->left = tmp;
    else
      node->parent->right = tmp;
  }

  tmp->parent = node->parent;
  node->parent = tmp;
  node->left = tmp->right;
  if (tmp->right != nullptr) tmp->right->parent = node;
  tmp->right = node;
}

void rb_tree::delete_node(int val) { delete_node(search(val)); }

/* 删除节点的三种情况
 * 1. 删除结点无子结点，直接删除
 * 2. 删除结点只有一个子结点，用子结点替换删除结点
 * 3. 删除结点有两个子结点，用后继结点（小于删除结点的最大结点）替换删除结点
 * 最后修复节点颜色
 */
void rb_tree::delete_node(rb_tree_base* node) {
  _rb_tree_color orgin_color = node->color;
  if (node == nullptr) return;
  // 如果存在父节点，将父节点的孩子节点置为空，并指示左右 -1 左 1 右 0 不存在
  rb_tree_base* node_parent = node->parent;
  int left = 0;
  // 卸载父节点
  if (node->parent != nullptr) {
    if (node->parent->left == node) {
      node->parent->left = nullptr;
      left = -1;
    } else {
      node->parent->right = nullptr;
      left = 1;
    }
  }

  // 对应下面的两种情况
  if (node->left != nullptr && node->right == nullptr) {
    // 节点的左孩子不为空 将此节点替换为左孩子节点
    node->left->parent = node_parent;
    if (left != 0 && left == -1)
      node_parent->left = node->left;
    else
      node_parent->right = node->left;

    node->left->parent = node_parent;
    node->left = nullptr;
  } else if (node->left == nullptr && node->right != nullptr) {
    // 节点的右孩子不为空
    node->left->parent = node_parent;
    if (left != 0 && left == -1)
      node_parent->left = node->right;
    else
      node_parent->right = node->right;

    node->right->parent = node_parent;
    node->right = nullptr;
  } else if (node->left != nullptr && node->right != nullptr) {
    // 节点的两个孩子都不为空 找到左子树中最大的孩子节点
    rb_tree_base* node_max = _find_max(node->left);

    auto node_max_color = node_max->color;

    // 置换根节点
    if (left == 0) _root = node_max;

    node_max->left = node->left;
    node_max->right = node->right;
    node_max->color = node->color;

    node->left->parent = node_max;
    node->right->parent = node_max;

    node->left = nullptr;
    node->right = nullptr;
    node->parent = node_max->parent;
    node->color = node_max_color;

    if (node_max->parent->left == node_max)
      node_max->parent->left = node;
    else
      node_max->parent->right = node;

    node_max->parent = node_parent;

    if (left != 0 && left == 1)
      node_parent->right = node_max;
    else if (left != 0 && left == -1)
      node_parent->left = node_max;
  }
  // 叶子节点，直接删除
  _delete_fixup(node, left, orgin_color);
}

/* 直接删除传入的节点，此节点一定是叶子节点
 * left != 0 传入节点的颜色为原始颜色
 * left == 0 传入节点的颜色为替换节点的颜色
 * orgin_color 表示要删除节点原来的颜色
 *
 * 1. 原来即是红色叶子节点 直接删除
 */
void rb_tree::_delete_fixup(rb_tree_base* node, int left, _rb_tree_color orgin_color) {
  // 删除当前节点，并向上修复红黑树
  // 卸载父节点 left != 0 已经在删除中卸载
  rb_tree_base* iter_node = node->parent;

  if (left == 0 && node->left == nullptr && node->right == nullptr && node->parent == nullptr)
    // 已经没有剩余节点
    _root = nullptr;
  else if (left == 0 && _root != nullptr) {
    // 当前为为根节点，但是有剩余
    if (iter_node->left == node)
      iter_node->left = nullptr;
    else
      iter_node->right = nullptr;
  }
  // 记录删除节点的颜色
  auto node_color = node->color;
  delete node;
  if (left != 0 && orgin_color == _rb_tree_color::_red && iter_node->left == nullptr &&
      iter_node->right == nullptr) {
    // 删除的节点为红色 且 是叶子节点
    return;
  } else if () {
  }

  // 向上修复颜色
  while (iter_node != nullptr) {
  }
}

/* 查找当前子树中最大的节点 */
rb_tree_base* rb_tree::_find_max(rb_tree_base* node) {
  while (node != nullptr) node = node->right;
  return node;
}

/* 查找当前子树中最小的节点 */
rb_tree_base* rb_tree::_find_min(rb_tree_base* node) {
  while (node != nullptr) node = node->left;
  return node;
}

/* 查找节点，找到即返回，找不到返回空 */
rb_tree_base* rb_tree::search(int value) {
  rb_tree_base* iter = _root;
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
 */
void rb_tree::pre_order_for() {
  auto iter_node = _root;
  std::stack<rb_tree_base*> nodes;
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
  std::stack<rb_tree_base*> nodes;
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
  rb_tree_base *iter_node = nullptr, *pre = nullptr;
  std::stack<rb_tree_base*> nodes;
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

void rb_tree::_pre_order(rb_tree_base* node) const {
  if (node == nullptr) return;
  std::cout << "  " << node->val << " | "
            << ((node->color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
  _pre_order(node->left);
  _pre_order(node->right);
}

void rb_tree::_in_order(rb_tree_base* node) const {
  if (node == nullptr) return;
  _in_order(node->left);
  std::cout << "  " << node->val << " | "
            << ((node->color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
  _in_order(node->right);
}

void rb_tree::_post_order(rb_tree_base* node) const {
  if (node == nullptr) return;
  _post_order(node->left);
  _post_order(node->right);
  std::cout << "  " << node->val << " | "
            << ((node->color == _rb_tree_color::_red) ? "red" : "black") << std::endl;
}

void rb_tree::bfs() {
  std::queue<rb_tree_base*> stack;
  stack.push(_root);

  while (!stack.empty()) {
    if (stack.front()->left != nullptr) stack.push(stack.front()->left);
    if (stack.front()->right != nullptr) stack.push(stack.front()->right);
    std::cout << "  " << stack.front()->val << " | "
              << (stack.front()->color == _rb_tree_color::_red ? "red" : "black") << std::endl;
    stack.pop();
  }
}

/* 节点的前序遍历非递归版本 */
void rb_tree::dfs() {
  std::stack<rb_tree_base*> stack;
  stack.push(_root);

  while (stack.empty() == false) {
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

/* 插入节点并修复新插入的节点 */
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

/* 插入节点的修复操作
 * 1. 更新节点的平衡因子
 * 2. 平衡因子为 0 表明新插入的节点不会导致树高度的增加 所以不需要向上传播
 * 3. 平衡因子为 1 或 -1 表明新插入的节点增加了树高，需要向上传播
 * 4. 平衡因子为 2 或 -2 表明新插入的节点已经导致树失衡 需要调整
 *    4.1 当前节点平衡因子为 2 子节点平衡因子为 1 则需要进行 LL旋转
 *    4.2 当前节点平衡因子为 2 子节点平衡因子为 -1 或 0 则需要进行 LR旋转
 *    4.3 当前节点平衡因子为 -2 子节点平衡因子为 -1 则需要进行 RR旋转
 *    4.4 当前节点平衡因子为 -2 子节点平衡因子为 1 或 0 则需要进行 RL旋转 */
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

/* 右右双螺旋 返回节点B
 *          A             B
 *           \           / \
 *            B         A  D
 *           / \       /
 *          C  D      C
 */
avl_tree_base* tree_avl::_rotate_rr(avl_tree_base* node) {
  avl_tree_base* ret = node->right;
  if (node->parent == nullptr) {
    // 如果当前节点为根节点
    _root = ret;
  } else {
    // 父节点挂载子树
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

/* 右左双螺旋 返回节点C
 *          A             C
 *           \           / \
 *           B          A  B
 *          / \             \
 *         C  D             D
 * */
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
  if (ret->left) ret->left->parent = node;

  tmp->parent = ret;
  tmp->left = ret->right;

  ret->left = node;
  ret->right = tmp;

  // 更正平衡因子
  _update_bf(ret);

  return ret;
}

/* 左左双螺旋 返回节点B
 *          A             B
 *         /             / \
 *        B    =>       C  A
 *       / \             \
 *      C  D             D
 */
avl_tree_base* tree_avl::_rotate_ll(avl_tree_base* node) {
  avl_tree_base* ret = node->left;
  if (node->parent == nullptr) {
    // 如果当前节点为根节点
    _root = ret;
  } else {
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

/* 左右双螺旋 返回节点D
 *          A             D
 *         /             / \
 *        B    =>       B  A
 *       / \           /
 *      C  D          C
 */
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
  if (ret->right) ret->right->parent = node;

  tmp->parent = ret;
  tmp->right = ret->left;

  ret->left = tmp;
  ret->right = node;

  // 更正平衡因子
  _update_bf(ret);

  return ret;
}

/* 更新旋转后的平衡因子 */
void tree_avl::_update_bf(avl_tree_base* node) {
  node->balance_factor = _get_height(node->left) - _get_height(node->right);
  node->left->balance_factor = _get_height(node->left->left) - _get_height(node->left->right);
  node->right->balance_factor = _get_height(node->right->left) - _get_height(node->right->right);
}

/* 获取整体树高 */
int tree_avl::get_height() { return _get_height(_root); }

/* 递归获取当前节点的树高 */
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
  avl_tree_base* node_parent = node->parent;
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
  else if (left == 0 && node->left == nullptr && node->right == nullptr && node->parent == nullptr)
    // 当前节点为根节点 且 没有剩余节点
    _root = nullptr;

  auto iter_node = node->parent;
  if (left == 0 && _root != nullptr) {
    // 当前节点为根节点 且 有剩余节点
    // 替换位置的节点 解除其父节点的挂载
    if (iter_node->left == node)
      iter_node->left = nullptr;
    else
      iter_node->right = nullptr;
  }

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

  for (auto nums_max = *std::max_element(nums.begin(), nums.end()),
            nums_min = *std::min_element(nums.begin(), nums.end());
       new_tree->get_height() != 0;) {
    int delete_num = (std::rand() % (nums_max - nums_min + 1)) + nums_min;
    new_tree->delete_val(delete_num);
  }

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

int main() {
  //  tiny_rb_tree::test::test_rb_tree();
  tiny_avl_tree::test::test_avl_tree();
  return 0;
}
