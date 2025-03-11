#include <stdio.h>
#include <stdlib.h>
int main() {
  char message[20] = "hello";
  printf("%s\n", message);
  printf("%ld\n", sizeof(message));

  int array[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  printf("%d\n", array[0]);

  int *array_p = (int *)malloc(array[5] * sizeof(int));
  printf("%ld\n", sizeof(*array_p));
  for (int i = 0; i < array[5]; i++) {
    array_p[i] = 2 * i;
  }
  for (int i = 0; i < array[5]; i++) {
    printf("%d,", array_p[i]);
  }
  printf("\n");
  return 0;
}
