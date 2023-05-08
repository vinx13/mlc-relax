#include <list>
#include <unordered_set>

namespace tvm {
namespace support {

template <typename T>
class OrderedSet {
 public:
  void push_back(const T& t) {
    if (!elem_to_iter_.count(t)) {
      elements_.push_back(t);
      elem_to_iter_[t] = std::prev(elements_.end());
    }
  }

  void erase(const T& t) {
    if (auto it = elem_to_iter_.find(t); it != elem_to_iter_.end()) {
      elements_.erase(it->second);
      elem_to_iter_.erase(it);
    }
  }

  void clear() {
    elements_.clear();
    elem_to_iter_.clear();
  }

  auto begin() const { return elements_.begin(); }
  auto end() const { return elements_.end(); }
  auto size() const { return elements_.size(); }
  auto empty() const { return elements_.empty(); }

 private:
  std::list<T> elements_;
  std::unordered_map<T, typename std::list<T>::iterator> elem_to_iter_;
};

}  // namespace support
}  // namespace tvm
