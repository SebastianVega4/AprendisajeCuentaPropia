import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int interacciones = Integer.parseInt(scanner.nextLine());

        // Procesamos cada interacción
        for (int i = 0; i < interacciones; i++) {
            int yearsTienen = Integer.parseInt(scanner.nextLine());
            String yearsWintSpaces = scanner.nextLine();
            String[] years = yearsWintSpaces.split(" ");

            int maxYear = Integer.parseInt(years[0]);
            int minYear = maxYear;

            // Buscar el máximo y mínimo de una sola pasada
            for (int j = 1; j < years.length; j++) {
                int yearInt = Integer.parseInt(years[j]);
                if (yearInt > maxYear) maxYear = yearInt;
                if (yearInt < minYear) minYear = yearInt;
            }

            // Calcular la cantidad de años faltantes
            int yearAlls = maxYear - minYear + 1;
            System.out.println(yearAlls - yearsTienen);
        }

        scanner.close();
    }
}