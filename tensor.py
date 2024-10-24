
class Tensor:
    
    def __init__(self, data):
        
        if not isinstance(data, list):
            raise TypeError("Tensor `data` must be a list.")
        
        self.data = data
        self.shape = self.get_shape(data)
        self.rank = self.get_rank(data)
        
    def get_shape(self, data):
        if isinstance(data, list):
            return  [len(data)] + (self.get_shape(data[0]) if len(data) > 0 else [])
        else:
            return []
    
    def get_rank(self, data):
        return len(data)
    
    def is_symmetric(self):
        """Sprawdzanie symetrii tensora o randze 2 (macierz)"""
        if self.rank != 2:
            raise ValueError("Symetria jest zdefiniowana tylko dla tensorów o randze 2")
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.data[i][j] != self.data[j][i]:
                    return False
        return True
    
    def transform(self, matrix):
        """Symulacja prostej transformacji tensora o randze 2 (macierz)"""
        if self.rank != 2:
            raise ValueError("Transformacje są zdefiniowane tylko dla tensorów o randze 2")
        
        transformed_tensor = self.multiply_matrices(matrix, self.data)
        return Tensor(transformed_tensor)
    
    def test_linearity(self, other, scalar):
        """Sprawdza liniowość tensora: (a*T1 + T2)"""
        if self.shape != other.shape:
            raise ValueError("Tensory muszą mieć ten sam kształt")
        
        # Sprawdzanie (a*T1 + T2)
        left = self * scalar + other
        right = Tensor(self._add(self._mul(self.data, scalar), other.data))
        
        return left.data == right.data
        
    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data}, rank={self.rank})"
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __add__(self, other):
        """Dodawanie dwóch tensorów o takim samym kształcie"""
        if self.shape != other.shape:
            raise ValueError("Tensors myst have the same shape to be added")
        return Tensor(self._add(self.data, other.data))
    
    def _add(self, a, b):
        """ Dodawanie rekurencyjne """
        if isinstance(a, list) and isinstance(b, list):
            return [self._add(x, y) for x, y in zip(a, b)]
        else:
            return a+b
    
    def __mul__(self, scalar):
        return Tensor(self._mul(self.data, scalar)) 

    def _mul(self, data, scalar):
        if isinstance(data, list):
            return [self._mul(x, scalar) for x in data]
        else:
            return data*scalar
        
    def multiply_matrices(self, matrix1, matrix2):
        """Ręczne mnożenie macierzy"""
        # Sprawdzamy, czy mnożenie jest możliwe
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("Liczba kolumn macierzy pierwszej musi być równa liczbie wierszy macierzy drugiej.")
        
        result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
        
        # Mnożenie macierzy
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        
        return result
    