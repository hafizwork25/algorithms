def insertion_sort(arr):
    """
    Insertion Sort Algorithm
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_array)
    sorted_array = insertion_sort(test_array.copy())
    print("Sorted array:", sorted_array)
