# Proiect SD

**Proiect realizat de:**  
- Popa Bogdan-Constantin  
- Popa Stefan Liviu-Octavian  
- Ghita Vladut  

## Sortări realizate:

### Obligatorii:
- Merge Sort  
- Radix Sort  
- Shell Sort  

### La alegere:
- Bitonic Sort  
- Heap Sort  
- Natural Merge Sort  

## Complexitate

Toate sortările implementate au o complexitate de **O(n log n)**, în afară de bitonic sort, care are o complexitate de **O(log^2 n)**.

## Testare și performanță

Testele folosite pentru a măsura viteza algoritmilor sunt generate aleatoriu, însă sunt **optimizate pentru fiecare algoritm** în parte, astfel încât să se poată observa diferențele de performanță atunci când datele de intrare sunt favorabile pentru metoda respectivă.

## Compararea algoritmilor de sortare

| Algoritm             | Caz Favorabil | Caz Mediu   | Caz Defavorabil | Memorie suplimentară | Stabilitate |
|----------------------|---------------|-------------|------------------|------------------------|-------------|
| Merge Sort           | O(n log n)    | O(n log n)  | O(n log n)       | O(n)                   | Stabil      |
| Radix Sort           | O(nk)         | O(nk)       | O(nk)            | O(n + k)               | Stabil      |
| Shell Sort           | O(n log n)    | O(n log² n) | O(n²)            | O(1)                   | Instabil    |
| Bitonic Sort         | O(log² n)     | O(log² n)   | O(log² n)        | O(n)                   | Instabil    |
| Heap Sort            | O(n log n)    | O(n log n)  | O(n log n)       | O(1)                   | Instabil    |
| Natural Merge Sort   | O(n)          | O(n log n)  | O(n log n)       | O(n)                   | Stabil      |

Prezentare:
https://www.canva.com/design/DAGjMinIK8Y/W4q3N9NhL0gdnjsVwRDRUQ/edit?utm_content=DAGjMinIK8Y&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
