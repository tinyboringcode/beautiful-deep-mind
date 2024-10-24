from tensor import Tensor
def test():
    # Test inicjalizacji tensora
    print("TEST 1: Inicjalizacja tensora")
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([[5, 6], [7, 8]])
    
    assert t1.shape == [2, 2], f"Expected shape [2, 2], got {t1.shape}"
    assert t2.shape == [2, 2], f"Expected shape [2, 2], got {t2.shape}"
    assert t1.rank == 2, f"Expected rank 2, got {t1.rank}"
    assert t1[0] == [1, 2], f"Expected first row to be [1, 2], got {t1[0]}"
    #print("PASSED\n")
    # Test dodawania tensorów
    print("TEST 2: Dodawanie tensorów")
    t3 = t1 + t2
    expected_sum = [[6, 8], [10, 12]]
    assert t3.data == expected_sum, f"Expected {expected_sum}, got {t3.data}"
    print("PASSED\n")
    
    # Test mnożenia przez skalar
    print("TEST 3: Mnożenie przez skalar")
    scalar = 2
    t4 = t1 * scalar
    expected_scalar_mul = [[2, 4], [6, 8]]
    assert t4.data == expected_scalar_mul, f"Expected {expected_scalar_mul}, got {t4.data}"
    print("PASSED\n")
    
    # Test symetrii
    print("TEST 4: Symetria tensora")
    symmetric_tensor = Tensor([[1, 2], [2, 1]])
    non_symmetric_tensor = Tensor([[1, 2], [3, 4]])
    
    assert symmetric_tensor.is_symmetric(), "Expected tensor to be symmetric"
    assert not non_symmetric_tensor.is_symmetric(), "Expected tensor to not be symmetric"
    print("PASSED\n")
    
    # Test mnożenia macierzy (transformacja)
    print("TEST 5: Mnożenie macierzy (transformacja)")
    matrix = [[1, 0], [0, 1]]  # Macierz jednostkowa
    t5 = t1.transform(matrix)   # Przekształcenie przez macierz jednostkową nie zmienia tensora
    
    assert t5.data == t1.data, f"Expected {t1.data}, got {t5.data}"
    print("PASSED\n")
    
    # Test liniowości
    print("TEST 6: Testowanie liniowości tensora")
    assert t1.test_linearity(t2, scalar), "Liniowość powinna być zachowana"
    print("PASSED\n")
    
    
    print(t1, "\n", t2, "\n", t3, "\n", t4, "\n", t5)
    print()
    print(t1.test_linearity(t2, scalar))
test()