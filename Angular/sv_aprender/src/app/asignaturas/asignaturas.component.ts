import { Component, input } from '@angular/core';

@Component({
  selector: 'app-asignaturas',
  imports: [],
  templateUrl: './asignaturas.component.html',
  styleUrl: './asignaturas.component.css'
})
export class AsignaturasComponent {
  materias = [
    {
      id: 1,
      materia: 'Calculo'
    },
    {
      id:2,
      materia: 'Programacion'
    },
    {
      id: 3,
      materia: 'Redes'
    }
  ]
}
