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

| Algoritm             | Caz Favorabil | Caz Mediu   | Caz Defavorabil | Memorie suplimentară | Caz optim                               | Caz problematic                              |
|----------------------|---------------|-------------|------------------|------------------------|------------------------------------------|----------------------------------------------|
| Merge Sort           | O(n log n)    | O(n log n)  | O(n log n)       | O(n)                   | Date deja parțial sortate                | Cost suplimentar de memorie, liste mari      |
| Radix Sort           | O(nk)         | O(nk)       | O(nk)            | O(n + k)               | Chei mici sau fixate, cum ar fi int-uri  | Chei lungi, variabile, sau non-numerice      |
| Shell Sort           | O(n log n)    | O(n log² n) | O(n²)            | O(1)                   | Liste mici, aproape sortate              | Liste mari, complet inversate                |
| Bitonic Sort         | O(log² n)     | O(log² n)   | O(log² n)        | O(n log n)             | Pe hardware paralelizabil (GPU, etc.)    | Ineficient pe procesoare secvențiale         |
| Heap Sort            | O(n log n)    | O(n log n)  | O(n log n)       | O(1)                   | Date nesortate, fără pattern specific    | Nu păstrează stabilitatea sortării           |
| Natural Merge Sort   | O(n)          | O(n log n)  | O(n log n)       | O(n)                   | Date cu secvențe deja ordonate (runs)    | Date complet aleatorii, fără ordine          |

