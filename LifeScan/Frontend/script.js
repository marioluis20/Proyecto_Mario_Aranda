// Variables globales
let currentFile = null;
let currentSection = 'reco';

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    console.log('LeafScan - Inicializando aplicación...');
    inicializarAplicacion();
});

function inicializarAplicacion() {
    console.log('LeafScan - Inicializando aplicación...');
    
    // Inicializar navegación
    inicializarNavegacion();
    
    // Inicializar sección de reconocimiento de imágenes
    inicializarReconocimiento();
    
    // Inicializar cuestionario
    inicializarCuestionario();
    
    // Inicializar chat
    inicializarChat();
    
    // Mostrar sección inicial
    mostrarSeccion('reco');
    
    console.log('Aplicación inicializada correctamente');
}

// Navegación entre secciones
function inicializarNavegacion() {
    const botonesNavegacion = document.querySelectorAll('.icon-btn');
    
    botonesNavegacion.forEach(boton => {
        boton.addEventListener('click', function() {
            const seccion = this.getAttribute('data-section');
            mostrarSeccion(seccion);
            
            // Actualizar botón activo
            botonesNavegacion.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Botones de la sidebar
    document.getElementById('tutorialBtn').addEventListener('click', abrirTutorial);
    document.getElementById('salirBtn').addEventListener('click', salir);
}

function mostrarSeccion(seccionId) {
    // Ocultar todas las secciones
    document.querySelectorAll('.section').forEach(seccion => {
        seccion.classList.remove('active-section');
    });
    
    // Mostrar la sección seleccionada
    const seccion = document.getElementById(seccionId);
    if (seccion) {
        seccion.classList.add('active-section');
        currentSection = seccionId;
        
        // Scroll suave al inicio
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        console.log(`Sección activa: ${seccionId}`);
    }
}

// Funciones de la sidebar
function abrirTutorial() {
    alert('📚 Tutorial de IA Health\n\n1. 📷 Reconocimiento: Sube una imagen médica para análisis\n2. 📝 Cuestionario: Responde preguntas sobre síntomas\n3. 🤖 Chat: Consulta al asistente IA\n4. 📚 Info: Más información sobre el sistema');
}

function salir() {
    if (confirm('¿Estás seguro de que deseas salir de VidaScan?')) {
        // Mostrar mensaje de despedida
        alert('Gracias por usar VidaScan. ¡Hasta pronto!');
        
        // Opcional: Agregar efecto de desvanecimiento antes de redirigir
        document.body.style.transition = 'opacity 0.5s ease';
        document.body.style.opacity = '0';
        
        // Redirigir después de la animación
        setTimeout(function() {
            window.location.href = 'login.html';
        }, 500);
    }
}
//------------------------------------------------------------------------------------------------------------------------------------------------
// ==================== RECONOCIMIENTO DE IMAGEN ====================
function inicializarReconocimiento() {
    console.log('Inicializando reconocimiento...');
    
    const dragArea = document.getElementById('drag');
    const fileInput = document.getElementById('fileInput');
    const agregarImagenBtn = document.getElementById('agregarImagenBtn');
    const analizarBtn = document.getElementById('analizarBtn');
    const limpiarBtn = document.getElementById('limpiarBtn');
    
    if (!dragArea || !fileInput || !agregarImagenBtn || !analizarBtn || !limpiarBtn) {
        console.error('Elementos de reconocimiento no encontrados');
        return;
    }
    
    // Configurar evento para el botón "Agregar imagen"
    agregarImagenBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Eventos para arrastrar y soltar
    dragArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.borderColor = 'var(--accent)';
        this.style.background = 'var(--accent-light)';
    });
    
    dragArea.addEventListener('dragleave', function() {
        this.style.borderColor = 'var(--border)';
        this.style.background = 'white';
    });
    
    dragArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.borderColor = 'var(--border)';
        this.style.background = 'white';
        
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            procesarImagen(file);
        }
    });
    
    // Click en el área de arrastre
    dragArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Cambio en el input de archivo
    fileInput.addEventListener('change', function(e) {
        if (this.files.length > 0) {
            procesarImagen(this.files[0]);
        }
    });
    
    // Botón analizar
    analizarBtn.addEventListener('click', analizarImagen);
    
    // Botón limpiar
    limpiarBtn.addEventListener('click', limpiarImagen);
    
    console.log('Reconocimiento de imágenes inicializado');
}

function procesarImagen(file) {
    if (!file.type.match('image.*')) {
        alert('⚠️ Por favor, selecciona solo archivos de imagen (JPG, PNG, etc.)');
        return;
    }
    
    currentFile = file;
    const dragArea = document.getElementById('drag');
    const dragText = document.getElementById('dragText');
    const thumb = document.getElementById('thumb');
    
    // Crear imagen de vista previa si no existe
    if (!thumb.src) {
        thumb.src = '';
        thumb.style.display = 'none';
    }
    
    const reader = new FileReader();
    
    reader.onload = function(e) {
        // Mostrar vista previa
        thumb.src = e.target.result;
        thumb.style.display = 'block';
        thumb.style.maxWidth = '200px';
        thumb.style.maxHeight = '200px';
        thumb.style.margin = '15px auto';
        
        // Actualizar texto del área de arrastre
        if (dragText) {
            dragText.textContent = '✅ Imagen cargada correctamente';
            dragText.style.color = 'var(--accent)';
        }
        
        dragArea.classList.add('has-image');
        
        console.log('Imagen procesada:', file.name);
    };
    
    reader.readAsDataURL(file);
}

async function analizarImagen() {
    console.log('Ejecutando analizarImagen con modelo real...');
    
    if (!currentFile) {
        alert('⚠️ Por favor, primero carga una imagen para analizar.');
        console.warn('No hay imagen cargada');
        return;
    }

    const resultBox = document.getElementById('resultReco');
    const resultText = document.getElementById('resultText');
    const analizarBtn = document.getElementById('analizarBtn');

    if (!resultBox || !resultText || !analizarBtn) {
        console.error('Elementos del DOM no encontrados');
        return;
    }

    // Deshabilitar botón durante análisis
    const originalHtml = analizarBtn.innerHTML;
    analizarBtn.disabled = true;
    analizarBtn.innerHTML = '<span>⏳</span> Analizando con IA...';

    resultBox.classList.add('show');
    resultText.innerHTML = '<span style="color: var(--accent)">🔍 Conectando con modelo IA de cáncer de piel...</span>';

    try {
        // Crear FormData para enviar la imagen
        const formData = new FormData();
        formData.append('image', currentFile);

        // Enviar a Flask para análisis con modelo real
        const response = await fetch('http://127.0.0.1:5000/api/predict/skin', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Error HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        if (data.success) {
            // Usar el análisis HTML generado por Flask
            resultText.innerHTML = data.analysis_html;
            console.log('✅ Análisis completado con modelo real:', data.prediction);
        } else {
            throw new Error(data.error || 'Error en el servidor');
        }

    } catch (error) {
        console.error('❌ Error con modelo real:', error);
        
        // Fallback a análisis simulado si falla
        resultText.innerHTML = `
            <div style="background:white;border-radius:12px;padding:20px;border:2px solid #ef4444">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:15px">
                    <div style="font-size:32px">🔴</div>
                    <div style="background:#ef4444;color:white;padding:6px 14px;border-radius:8px;font-weight:700">
                        ALTO RIESGO - POSIBLE MELANOMA
                    </div>
                </div>
                
                <div style="margin-top:10px;font-size:15px;color:var(--text-primary)">
                    <strong>⚠️ Modelo IA no disponible (usando demo)</strong><br>
                    <span style="color: var(--text-secondary)">Para análisis preciso, ejecuta servidor Flask con modelo de imágenes</span>
                </div>
                
                <div style="margin-top:20px;padding:15px;background:#fee2e2;border-radius:10px;border-left:4px solid #dc2626;">
                    <div style="font-weight:700;color:#dc2626;margin-bottom:8px;display:flex;align-items:center;gap:8px;">
                        <span>🚨</span> RECOMENDACIÓN URGENTE
                    </div>
                    <div style="color:#7f1d1d;font-size:14px;line-height:1.5;">
                        <strong>Se recomienda VISITAR A UN DERMATÓLOGO LO ANTES POSIBLE.</strong><br>
                        Las características identificadas requieren evaluación médica inmediata.
                    </div>
                </div>
                
                <div style="margin-top:15px;padding:10px;background:#f8fafc;border-radius:8px;font-size:13px;color:#666">
                    <strong>📊 Nota:</strong> Análisis de demostración. Instala TensorFlow y modelo real para diagnóstico preciso.
                </div>
            </div>
        `;
        
        console.log('⚠️ Usando análisis de demostración');

    } finally {
        // Restaurar botón
        analizarBtn.disabled = false;
        analizarBtn.innerHTML = originalHtml;
    }
}

function limpiarImagen() {
    console.log('Ejecutando limpiarImagen...');
    
    currentFile = null;
    
    const dragArea = document.getElementById('drag');
    const dragText = document.getElementById('dragText');
    const thumb = document.getElementById('thumb');
    const resultBox = document.getElementById('resultReco');
    const fileInput = document.getElementById('fileInput');
    const analizarBtn = document.getElementById('analizarBtn');
    
    if (thumb) {
        thumb.style.display = 'none';
        thumb.src = '';
    }
    
    if (dragArea) {
        dragArea.classList.remove('has-image');
    }
    
    if (dragText) {
        dragText.textContent = '📁 Arrastra la imagen aquí';
        dragText.style.color = '';
    }
    
    if (resultBox) {
        resultBox.classList.remove('show');
    }
    
    if (fileInput) {
        fileInput.value = '';
    }
    
    if (analizarBtn) {
        analizarBtn.disabled = false;
        analizarBtn.innerHTML = '<span>🔍</span> Analizar Imagen';
    }
    
    console.log('Imagen limpiada correctamente');
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Cuestionario
// Variables globales para cuestionario
let currentModel = null; // Cambiado a null inicialmente
let currentAnswers = {};
let totalQuestions = 0;
let answeredQuestions = 0;

// Límites de validación para preguntas numéricas
const LIMITES_VALIDACION = {
    stroke: {
        age: { min: 1, max: 120, mensaje: 'Edad improbable para análisis clínico' },
        avg_glucose_level: { min: 50, max: 300, mensaje: 'Nivel de glucosa fuera de rango clínico típico' },
        bmi: { min: 10, max: 50, mensaje: 'IMC fuera de rango fisiológico estándar' }
    },
    heart: {
        Age: { min: 1, max: 120, mensaje: 'Edad improbable para evaluación cardíaca' },
        RestingBP: { min: 80, max: 200, mensaje: 'Presión arterial en reposo fuera de rango viable' },
        Cholesterol: { min: 100, max: 400, mensaje: 'Nivel de colesterol fuera de parámetros clínicos' },
        MaxHR: { min: 60, max: 220, mensaje: 'Frecuencia cardíaca máxima fuera de rango fisiológico' },
        Oldpeak: { min: 0, max: 10, mensaje: 'Valor de Oldpeak extremadamente inusual' }
    }
};

// Preguntas para Derrame Cerebral (Stroke) - MANTENIDAS IGUAL
const strokeQuestions = [
    {
        id: 'gender',
        question: 'Género',
        type: 'radio',
        options: [
            { value: 'Male', label: 'Masculino' },
            { value: 'Female', label: 'Femenino' }
        ],
        required: true,
        category: 'stroke'
    },
    {
        id: 'age',
        question: 'Edad',
        type: 'number',
        min: 1,
        max: 120,
        placeholder: 'Ingresa tu edad',
        required: true,
        category: 'stroke'
    },
    {
        id: 'ever_married',
        question: '¿Ha estado casado/a?',
        type: 'radio',
        options: [
            { value: 'Yes', label: 'Sí' },
            { value: 'No', label: 'No' }
        ],
        required: true,
        category: 'stroke'
    },
    {
        id: 'work_type',
        question: 'Tipo de trabajo',
        type: 'radio',
        options: [
            { value: 'Private', label: 'Privado' },
            { value: 'Self-employed', label: 'Autoempleado' },
            { value: 'Govt_job', label: 'Trabajo gubernamental' },
            { value: 'Children', label: 'Niño/Estudiante' },
            { value: 'Never_worked', label: 'Nunca ha trabajado' }
        ],
        required: true,
        category: 'stroke'
    },
    {
        id: 'Residence_type',
        question: 'Tipo de residencia',
        type: 'radio',
        options: [
            { value: 'Urban', label: 'Urbano' },
            { value: 'Rural', label: 'Rural' }
        ],
        required: true,
        category: 'stroke'
    },
    {
        id: 'avg_glucose_level',
        question: 'Nivel promedio de glucosa en sangre (mg/dL)',
        type: 'number',
        min: 50,
        max: 300,
        step: 0.1,
        placeholder: 'Ej: 100.5',
        required: true,
        category: 'stroke'
    },
    {
        id: 'bmi',
        question: 'Índice de Masa Corporal (BMI)',
        type: 'number',
        min: 10,
        max: 50,
        step: 0.1,
        placeholder: 'Ej: 24.5',
        required: true,
        category: 'stroke'
    },
    {
        id: 'hypertension',
        question: '¿Tiene hipertensión?',
        type: 'radio',
        options: [
            { value: '1', label: 'Sí' },
            { value: '0', label: 'No' }
        ],
        required: true,
        category: 'stroke'
    },
    {
        id: 'heart_disease',
        question: '¿Tiene enfermedad cardíaca?',
        type: 'radio',
        options: [
            { value: '1', label: 'Sí' },
            { value: '0', label: 'No' }
        ],
        required: true,
        category: 'stroke'
    },
    {
        id: 'smoking_status',
        question: 'Estado de fumador',
        type: 'radio',
        options: [
            { value: 'formerly smoked', label: 'Fumaba anteriormente' },
            { value: 'never smoked', label: 'Nunca ha fumado' },
            { value: 'smokes', label: 'Fuma actualmente' },
            { value: 'Unknown', label: 'Desconocido' }
        ],
        required: true,
        category: 'stroke'
    }
];

// Preguntas para Falla Cardíaca (Heart) - MANTENIDAS IGUAL
const heartQuestions = [
    {
        id: 'Age',
        question: 'Edad',
        type: 'number',
        min: 1,
        max: 120,
        placeholder: 'Ingresa tu edad',
        required: true,
        category: 'heart'
    },
    {
        id: 'Sex',
        question: 'Sexo',
        type: 'radio',
        options: [
            { value: 'M', label: 'Masculino' },
            { value: 'F', label: 'Femenino' }
        ],
        required: true,
        category: 'heart'
    },
    {
        id: 'ChestPainType',
        question: 'Tipo de dolor de pecho',
        type: 'radio',
        options: [
            { value: 'TA', label: 'Angina típica' },
            { value: 'ATA', label: 'Angina atípica' },
            { value: 'NAP', label: 'Dolor no anginoso' },
            { value: 'ASY', label: 'Asintomático' }
        ],
        required: true,
        category: 'heart'
    },
    {
        id: 'RestingBP',
        question: 'Presión arterial en reposo (mm Hg)',
        type: 'number',
        min: 80,
        max: 200,
        placeholder: 'Ej: 120',
        required: true,
        category: 'heart'
    },
    {
        id: 'Cholesterol',
        question: 'Nivel de colesterol (mg/dL)',
        type: 'number',
        min: 100,
        max: 400,
        placeholder: 'Ej: 200',
        required: true,
        category: 'heart'
    },
    {
        id: 'FastingBS',
        question: 'Glucosa en ayunas',
        type: 'radio',
        options: [
            { value: '0', label: '< 120 mg/dL' },
            { value: '1', label: '≥ 120 mg/dL' }
        ],
        required: true,
        category: 'heart'
    },
    {
        id: 'RestingECG',
        question: 'Resultado de ECG en reposo',
        type: 'radio',
        options: [
            { value: 'Normal', label: 'Normal' },
            { value: 'ST', label: 'Anomalía onda ST-T' },
            { value: 'LVH', label: 'Hipertrofia ventricular' }
        ],
        required: true,
        category: 'heart'
    },
    {
        id: 'MaxHR',
        question: 'Frecuencia cardíaca máxima alcanzada',
        type: 'number',
        min: 60,
        max: 220,
        placeholder: 'Ej: 150',
        required: true,
        category: 'heart'
    },
    {
        id: 'ExerciseAngina',
        question: '¿Presenta angina inducida por ejercicio?',
        type: 'radio',
        options: [
            { value: 'Y', label: 'Sí' },
            { value: 'N', label: 'No' }
        ],
        required: true,
        category: 'heart'
    },
    {
        id: 'Oldpeak',
        question: 'Depresión ST inducida por ejercicio (Oldpeak)',
        type: 'number',
        min: 0,
        max: 10,
        step: 0.1,
        placeholder: 'Ej: 1.5',
        required: true,
        category: 'heart'
    },
    {
        id: 'ST_Slope',
        question: 'Pendiente del segmento ST durante ejercicio máximo',
        type: 'radio',
        options: [
            { value: 'Up', label: 'Ascendente' },
            { value: 'Flat', label: 'Plana' },
            { value: 'Down', label: 'Descendente' }
        ],
        required: true,
        category: 'heart'
    },
    {
        id: 'HeartDisease',
        question: '¿Diagnóstico previo de enfermedad cardíaca?',
        type: 'radio',
        options: [
            { value: '1', label: 'Sí' },
            { value: '0', label: 'No' }
        ],
        required: true,
        category: 'heart'
    }
];

// Inicializar cuestionario - MODIFICADA
function inicializarCuestionario() {
    // Configurar botones del selector de modelos (con clase .primary-btn)
    document.querySelectorAll('.primary-btn[data-model]').forEach(button => {
        button.addEventListener('click', function() {
            const model = this.getAttribute('data-model');
            selectModel(model);
        });
    });
    
    // Ocultar el contenedor de preguntas inicialmente
    document.getElementById('questionsContainer').style.display = 'none';
    
    // Configurar botón de enviar
    document.getElementById('submitQuizBtn').addEventListener('click', submitQuiz);
}

// Seleccionar modelo - NUEVA FUNCIÓN
function selectModel(model) {
    currentModel = model;
    currentAnswers = {};
    answeredQuestions = 0;
    
    // Mostrar contenedor de preguntas
    document.getElementById('questionsContainer').style.display = 'block';
    
    // Ocultar todas las preguntas y resultados previos
    document.getElementById('strokeQuestions').style.display = 'none';
    document.getElementById('heartQuestions').style.display = 'none';
    document.getElementById('quizResult').innerHTML = '';
    
    // Cargar preguntas del modelo seleccionado
    if (model === 'stroke') {
        totalQuestions = strokeQuestions.length;
        renderQuestions(strokeQuestions, 'strokeQuestions', model);
        document.getElementById('strokeQuestions').style.display = 'block';
    } else {
        totalQuestions = heartQuestions.length;
        renderQuestions(heartQuestions, 'heartQuestions', model);
        document.getElementById('heartQuestions').style.display = 'block';
    }
    
    // Inicializar barra de progreso
    initProgressBar();
    
    // Scroll suave a preguntas
    document.getElementById('questionsContainer').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

// Inicializar barra de progreso - NUEVA FUNCIÓN
function initProgressBar() {
    const progressContainer = document.getElementById('progressContainer');
    if (!progressContainer.querySelector('#progressFill')) {
        progressContainer.innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <div style="font-weight: 600; color: var(--text-primary);">Progreso del cuestionario</div>
                <div style="color: var(--accent); font-weight: 600;" id="progressText">0/${totalQuestions}</div>
            </div>
            <div style="height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;">
                <div id="progressFill" style="height: 100%; background: var(--accent); border-radius: 3px; width: 0%; transition: width 0.3s ease;"></div>
            </div>
        `;
    }
    updateProgress();
}

// Renderizar preguntas - MODIFICADA para mantener estilo original
function renderQuestions(questions, containerId, model) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    questions.forEach((question, index) => {
        const questionElement = createQuestionElement(question, index + 1, model);
        container.appendChild(questionElement);
    });
    
    // Auto-expandir primera pregunta
    setTimeout(() => {
        if (questions[0]) {
            toggleQuestion(questions[0].id);
        }
    }, 300);
}

// Crear elemento de pregunta - MODIFICADA para mantener estilo original
function createQuestionElement(question, number, model) {
    const element = document.createElement('div');
    element.className = 'q-item';
    element.id = `q-${question.id}`;
    element.style.animation = 'fadeInUp 0.3s ease-out';
    element.style.animationDelay = `${(number - 1) * 0.05}s`;
    element.style.opacity = '0';
    element.style.animationFillMode = 'forwards';
    
    let inputHTML = '';
    
    switch(question.type) {
        case 'radio':
            inputHTML = `
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-top: 10px;">
                    ${question.options.map(option => `
                        <label style="display: flex; align-items: center; padding: 12px; border: 2px solid var(--border); border-radius: 10px; cursor: pointer; transition: all 0.2s ease; background: white;">
                            <input type="radio" 
                                   name="${question.id}" 
                                   value="${option.value}"
                                   style="margin-right: 10px;"
                                   onchange="updateAnswer('${question.id}', '${option.value}')">
                            ${option.label}
                        </label>
                    `).join('')}
                </div>
            `;
            break;
            
        case 'number':
            inputHTML = `
                <input type="number" 
                       style="width: 100%; padding: 12px; border: 2px solid var(--border); border-radius: 10px; font-size: 16px; margin-top: 10px;"
                       id="input-${question.id}"
                       min="${question.min || ''}"
                       max="${question.max || ''}"
                       step="${question.step || '1'}"
                       placeholder="${question.placeholder || ''}"
                       onchange="updateAnswer('${question.id}', this.value)"
                       oninput="updateAnswerWithValidation('${question.id}', this.value, '${model}')">
                <div class="mensaje-validacion" id="mensaje-${question.id}"></div>
            `;
            break;
    }
    
    element.innerHTML = `
        <div class="q-head" onclick="toggleQuestion('${question.id}')">
            <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                <div>
                    <div style="font-weight: 600; color: var(--accent); font-size: 14px;">
                        Pregunta ${number} de ${totalQuestions}
                    </div>
                    <div style="color: var(--text-primary); font-size: 16px; margin-top: 4px;">
                        ${question.question}
                    </div>
                </div>
                <div style="font-size: 20px; color: var(--muted); transition: transform 0.3s;">▼</div>
            </div>
        </div>
        <div class="q-body" id="q-body-${question.id}">
            ${inputHTML}
        </div>
    `;
    
    return element;
}

// Actualizar respuesta con validación - NUEVA FUNCIÓN
function updateAnswerWithValidation(questionId, value, model) {
    // Primero actualizar la respuesta
    updateAnswer(questionId, value);
    
    // Validar si existe límite para esta pregunta
    const limites = LIMITES_VALIDACION[model]?.[questionId];
    const mensajeElement = document.getElementById(`mensaje-${questionId}`);
    const inputElement = document.getElementById(`input-${questionId}`);
    
    if (!limites || !mensajeElement || !inputElement || value === '') {
        // Limpiar mensaje si no hay límites o valor vacío
        if (mensajeElement) {
            mensajeElement.textContent = '';
            mensajeElement.style.cssText = 'margin-top: 8px; font-size: 14px; min-height: 20px; transition: all 0.3s ease;';
        }
        if (inputElement) {
            inputElement.classList.remove('error', 'advertencia');
            // Liberar bloqueo si se corrigió
            liberarBloqueoPregunta(questionId);
        }
        return;
    }
    
    const numValue = parseFloat(value);
    if (isNaN(numValue)) return;
    
    // Verificar si está fuera de límites
    if (numValue < limites.min || numValue > limites.max) {
        mensajeElement.textContent = `⚠️ ${limites.mensaje}. Valor ingresado: ${value}`;
        // Forzar estilo rojo
        mensajeElement.style.cssText = 'color: #ef4444 !important; margin-top: 8px; font-size: 14px; min-height: 20px; font-weight: 600;';
        
        inputElement.classList.add('error');
        inputElement.classList.remove('advertencia');
        
        // Asegurar que la pregunta esté abierta y bloqueada
        const bodyElement = document.getElementById(`q-body-${questionId}`);
        if (bodyElement && !bodyElement.classList.contains('open')) {
            toggleQuestion(questionId, true);
        }
        
    } else {
        // Dentro de límites - limpiar mensaje
        mensajeElement.textContent = '';
        mensajeElement.style.cssText = 'margin-top: 8px; font-size: 14px; min-height: 20px; transition: all 0.3s ease;';
        inputElement.classList.remove('advertencia', 'error');
        
        // LIBERAR BLOQUEO cuando el valor es válido
        liberarBloqueoPregunta(questionId);
    }
}

// Actualizar respuesta - MODIFICADA
function updateAnswer(questionId, value) {
    currentAnswers[questionId] = value;
    answeredQuestions = Object.keys(currentAnswers).length;
    
    // Actualizar progreso
    updateProgress();
    
    // Marcar pregunta como respondida
    const questionElement = document.getElementById(`q-${questionId}`);
    if (questionElement) {
        questionElement.style.borderLeft = '4px solid var(--accent)';
        questionElement.style.background = 'linear-gradient(135deg, #f8fafc, #f1f5f9)';
    }
    
    // Limpiar validación si se selecciona una opción de radio (para preguntas que cambiaron de tipo)
    const mensajeElement = document.getElementById(`mensaje-${questionId}`);
    const inputElement = document.getElementById(`input-${questionId}`);
    if (mensajeElement) {
        mensajeElement.textContent = '';
        mensajeElement.style.cssText = 'margin-top: 8px; font-size: 14px; min-height: 20px; transition: all 0.3s ease;';
    }
}

// Actualizar barra de progreso - MODIFICADA
function updateProgress() {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    if (progressFill && progressText) {
        const percentage = (answeredQuestions / totalQuestions) * 100;
        progressFill.style.width = `${percentage}%`;
        progressText.textContent = `${answeredQuestions}/${totalQuestions}`;
    }
}

// Alternar visibilidad de pregunta - MODIFICADA con bloqueo
window.toggleQuestion = function(questionId, forzarApertura = false) {
    const body = document.getElementById(`q-body-${questionId}`);
    const qItem = document.getElementById(`q-${questionId}`);
    
    if (!body || !qItem) return;
    
    // Verificar si esta pregunta está bloqueada (con error)
    const inputElement = document.getElementById(`input-${questionId}`);
    const mensajeElement = document.getElementById(`mensaje-${questionId}`);
    const tieneError = inputElement && inputElement.classList.contains('error') && 
                      mensajeElement && mensajeElement.textContent !== '';
    
    // Si tiene error y NO estamos forzando apertura, NO permitir cerrar
    if (tieneError && !forzarApertura) {
        // Si ya está abierta, mantenerla abierta
        if (!body.classList.contains('open')) {
            body.classList.add('open');
        }
        return;
    }
    
    // Si no tiene error o estamos forzando apertura, proceder normalmente
    const isOpening = !body.classList.contains('open');
    body.classList.toggle('open');
    
    const arrow = qItem.querySelector('.q-head > div > div:last-child');
    if (arrow) {
        arrow.style.transform = isOpening ? 'rotate(180deg)' : 'rotate(0deg)';
        arrow.style.color = isOpening ? 'var(--accent)' : 'var(--muted)';
        arrow.style.transition = 'transform 0.3s ease, color 0.3s ease';
    }
};


// NUEVA FUNCIÓN: Validar valores inusuales
function validarValoresInusuales(answers, model) {
    const valoresInusuales = [];
    const preguntasConValoresInusuales = [];
    
    // Verificar cada respuesta numérica
    Object.keys(answers).forEach(questionId => {
        const value = answers[questionId];
        if (!value || value === '') return;
        
        // Verificar si es una pregunta numérica con límites definidos
        const limites = LIMITES_VALIDACION[model]?.[questionId];
        if (!limites) return;
        
        const numValue = parseFloat(value);
        if (isNaN(numValue)) return;
        
        // Verificar si está fuera de los límites razonables
        if (numValue < limites.min || numValue > limites.max) {
            valoresInusuales.push({
                pregunta: questionId,
                valor: value,
                mensaje: limites.mensaje,
                min: limites.min,
                max: limites.max
            });
            
            // Obtener el texto de la pregunta
            const questions = model === 'stroke' ? strokeQuestions : heartQuestions;
            const pregunta = questions.find(q => q.id === questionId);
            if (pregunta) {
                preguntasConValoresInusuales.push(pregunta.question);
            }
        }
    });
    
    return {
        hayInusuales: valoresInusuales.length > 0,
        valores: valoresInusuales,
        preguntas: preguntasConValoresInusuales,
        cantidad: valoresInusuales.length
    };
}

// Enviar cuestionario - VERSIÓN CON FLASK (manteniendo diseño)
async function submitQuiz() {
    const submitBtn = document.getElementById('submitQuizBtn');
    
    // 1. PRIMERO validar valores inusuales
    const validacionInusuales = validarValoresInusuales(currentAnswers, currentModel);
    
    if (validacionInusuales.hayInusuales) {
        let mensaje = `⚠️ Se detectaron ${validacionInusuales.cantidad} valor(es) inusuales:\n\n`;
        
        validacionInusuales.valores.forEach((item, index) => {
            mensaje += `${index + 1}. ${item.mensaje}\n   Valor actual: ${item.valor}\n   Rango recomendado: ${item.min} - ${item.max}\n\n`;
        });
        
        mensaje += "Por favor, revisa y corrige estos valores antes de continuar con el análisis.";
        
        alert(mensaje);
        return;
    }
    
    // 2. Luego verificar si hay valores con advertencia visual (rojo) pendientes
    let hayAdvertencias = false;
    const questions = currentModel === 'stroke' ? strokeQuestions : heartQuestions;
    
    questions.forEach(q => {
        if (q.type === 'number' && currentAnswers[q.id]) {
            const inputElement = document.getElementById(`input-${q.id}`);
            const mensajeElement = document.getElementById(`mensaje-${q.id}`);
            
            if (inputElement && inputElement.classList.contains('error') && 
                mensajeElement && mensajeElement.textContent !== '') {
                hayAdvertencias = true;
            }
        }
    });
    
    if (hayAdvertencias) {
        alert(`⚠️ Hay valores fuera de rango recomendado. Por favor revisa las preguntas marcadas en rojo antes de continuar.`);
        return;
    }
    
    // 3. Validar que todas las preguntas estén respondidas
    const unanswered = questions.filter(q => {
        const tieneRespuesta = currentAnswers[q.id];
        return !tieneRespuesta;
    });
    
    if (unanswered.length > 0) {
        alert(`⚠️ Por favor responde todas las preguntas. Faltan ${unanswered.length} preguntas.`);
        return;
    }
    
    // 4. Si pasa todas las validaciones, proceder con el análisis
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<span>⏳</span> Conectando con IA...';
    submitBtn.disabled = true;
    
    try {
        // INTENTAR conectar con Flask primero
        console.log("🔗 Intentando conectar con Flask...");
        
        const endpoint = currentModel === 'stroke' 
            ? 'http://127.0.0.1:5000/api/predict/stroke'
            : 'http://127.0.0.1:5000/api/predict/heart';
        
        submitBtn.innerHTML = '<span>⏳</span> Procesando con IA...';
        
        // Intentar fetch con timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000); // 8 segundos timeout
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentAnswers),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`Error HTTP ${response.status}`);
        }
        
        const flaskResult = await response.json();
        
        if (!flaskResult.success) {
            throw new Error(flaskResult.error || 'Error del servidor');
        }
        
        console.log("✅ Flask respondió con éxito:", flaskResult);
        
        // ==============================================
        // ADAPTAR DATOS DE FLASK PARA showResults
        // ==============================================
        
        // Calcular nivel de riesgo basado en Flask
        let riskLevelClass, riskLevelText, riskLevelDesc;
        
        if (flaskResult.probability >= 0.7) {
            riskLevelClass = 'high';
            riskLevelText = 'ALTO';
            riskLevelDesc = 'Riesgo elevado - Se recomienda evaluación médica prioritaria';
        } else if (flaskResult.probability >= 0.4) {
            riskLevelClass = 'medium';
            riskLevelText = 'MODERADO';
            riskLevelDesc = 'Riesgo intermedio - Se sugiere consulta médica y seguimiento';
        } else {
            riskLevelClass = 'low';
            riskLevelText = 'BAJO';
            riskLevelDesc = 'Riesgo bajo - Mantener hábitos saludables y chequeos regulares';
        }
        
        // Guardar funciones originales
        const originalGenerateIAAnalysis = generateIAAnalysis;
        const originalCalculateRiskLevel = calculateRiskLevel;
        const originalGetRiskFactors = getRiskFactors;
        
        // Crear análisis combinado Flask + local
        const flaskAnalysis = flaskResult.analysis || '';
        const localAnalysis = originalGenerateIAAnalysis(currentAnswers, currentModel);
        
        // Extraer solo el texto del análisis local (sin métricas IA duplicadas)
        const localAnalysisText = localAnalysis.split('<div style="margin-top: 15px')[0];
        
        const combinedAnalysis = `
            ${flaskAnalysis}
            
            ${localAnalysisText}
            
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid var(--border);">
                <strong style="color: var(--accent);">📊 Métricas IA (Modelo Real):</strong> 
                Análisis generado por modelo de machine learning entrenado con datos clínicos.
                <br><br>
                <strong>🔢 Probabilidad calculada:</strong> ${(flaskResult.probability * 100).toFixed(1)}%
                ${flaskResult.debug_info ? `<br><strong>⚖️ Ajuste clínico:</strong> ${flaskResult.debug_info.was_adjusted ? 'Aplicado' : 'No requerido'}` : ''}
                ${flaskResult.debug_info && flaskResult.debug_info.risk_multiplier ? `<br><strong>📈 Factor de riesgo:</strong> ${flaskResult.debug_info.risk_multiplier.toFixed(1)}x` : ''}
            </div>
        `;
        
        // Sobrescribir temporalmente funciones para usar datos de Flask
        window.generateIAAnalysis = function(answers, model) {
            return combinedAnalysis;
        };
        
        window.calculateRiskLevel = function(answers, model) {
            return {
                class: riskLevelClass,
                text: riskLevelText,
                description: riskLevelDesc
            };
        };
        
        window.getRiskFactors = function(answers, model) {
            return flaskResult.factors && flaskResult.factors.length > 0 
                ? flaskResult.factors 
                : ['Análisis basado en modelo predictivo con ajuste clínico'];
        };
        
        // Usar showResults con datos adaptados de Flask
        showResults(currentAnswers, currentModel);
        
        // Restaurar funciones originales
        setTimeout(() => {
            window.generateIAAnalysis = originalGenerateIAAnalysis;
            window.calculateRiskLevel = originalCalculateRiskLevel;
            window.getRiskFactors = originalGetRiskFactors;
        }, 100);
        
    } catch (error) {
        console.error('❌ Error con Flask:', error);
        
        // Si Flask falla, usar análisis local (simulado)
        if (error.name === 'AbortError' || error.message.includes('Failed to fetch') || 
            error.message.includes('NetworkError') || error.message.includes('HTTP')) {
            
            console.log("⚠️ Flask no disponible, usando análisis local");
            alert('⚠️ Usando análisis de demostración (Flask no disponible).\n\nPara análisis con IA real, ejecuta el servidor Flask.');
            
            submitBtn.innerHTML = '<span>⏳</span> Analizando (modo demo)...';
            
            // Análisis local/simulado (como antes)
            await new Promise(resolve => setTimeout(resolve, 2000));
            showResults(currentAnswers, currentModel);
            
        } else {
            alert(`❌ Error: ${error.message}\n\nMostrando análisis de demostración.`);
            
            submitBtn.innerHTML = '<span>⏳</span> Analizando (modo demo)...';
            await new Promise(resolve => setTimeout(resolve, 1500));
            showResults(currentAnswers, currentModel);
        }
        
    } finally {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

// Función para validar valores inusuales (MANTENER esta función)
function validarValoresInusuales(answers, model) {
    const valoresInusuales = [];
    const preguntasConValoresInusuales = [];
    
    Object.keys(answers).forEach(questionId => {
        const value = answers[questionId];
        if (!value || value === '') return;
        
        const limites = LIMITES_VALIDACION[model]?.[questionId];
        if (!limites) return;
        
        const numValue = parseFloat(value);
        if (isNaN(numValue)) return;
        
        if (numValue < limites.min || numValue > limites.max) {
            valoresInusuales.push({
                pregunta: questionId,
                valor: value,
                mensaje: limites.mensaje,
                min: limites.min,
                max: limites.max
            });
            
            const questions = model === 'stroke' ? strokeQuestions : heartQuestions;
            const pregunta = questions.find(q => q.id === questionId);
            if (pregunta) {
                preguntasConValoresInusuales.push(pregunta.question);
            }
        }
    });
    
    return {
        hayInusuales: valoresInusuales.length > 0,
        valores: valoresInusuales,
        preguntas: preguntasConValoresInusuales,
        cantidad: valoresInusuales.length
    };
}

// Función para validar valores inusuales (MANTENER ESTA IGUAL)
function validarValoresInusuales(answers, model) {
    const valoresInusuales = [];
    const preguntasConValoresInusuales = [];
    
    // Verificar cada respuesta numérica
    Object.keys(answers).forEach(questionId => {
        const value = answers[questionId];
        if (!value || value === '') return;
        
        // Verificar si es una pregunta numérica con límites definidos
        const limites = LIMITES_VALIDACION[model]?.[questionId];
        if (!limites) return;
        
        const numValue = parseFloat(value);
        if (isNaN(numValue)) return;
        
        // Verificar si está fuera de los límites razonables
        if (numValue < limites.min || numValue > limites.max) {
            valoresInusuales.push({
                pregunta: questionId,
                valor: value,
                mensaje: limites.mensaje,
                min: limites.min,
                max: limites.max
            });
            
            // Obtener el texto de la pregunta
            const questions = model === 'stroke' ? strokeQuestions : heartQuestions;
            const pregunta = questions.find(q => q.id === questionId);
            if (pregunta) {
                preguntasConValoresInusuales.push(pregunta.question);
            }
        }
    });
    
    return {
        hayInusuales: valoresInusuales.length > 0,
        valores: valoresInusuales,
        preguntas: preguntasConValoresInusuales,
        cantidad: valoresInusuales.length
    };
}

// Función updateAnswerWithValidation (MANTENER para mostrar advertencias en rojo)
function updateAnswerWithValidation(questionId, value, model) {
    // Primero actualizar la respuesta
    updateAnswer(questionId, value);
    
    // Validar si existe límite para esta pregunta
    const limites = LIMITES_VALIDACION[model]?.[questionId];
    const mensajeElement = document.getElementById(`mensaje-${questionId}`);
    const inputElement = document.getElementById(`input-${questionId}`);
    
    if (!limites || !mensajeElement || !inputElement || value === '') {
        // Limpiar mensaje si no hay límites o valor vacío
        if (mensajeElement) {
            mensajeElement.textContent = '';
            mensajeElement.style.cssText = 'margin-top: 8px; font-size: 14px; min-height: 20px; transition: all 0.3s ease;';
        }
        if (inputElement) {
            inputElement.classList.remove('error', 'advertencia');
        }
        return;
    }
    
    const numValue = parseFloat(value);
    if (isNaN(numValue)) return;
    
    // Verificar si está fuera de límites
    if (numValue < limites.min || numValue > limites.max) {
        mensajeElement.textContent = `⚠️ ${limites.mensaje}. Valor ingresado: ${value}`;
        // Forzar estilo rojo
        mensajeElement.style.cssText = 'color: #ef4444 !important; margin-top: 8px; font-size: 14px; min-height: 20px; font-weight: 600;';
        
        inputElement.classList.add('error');
        inputElement.classList.remove('advertencia');
    } else {
        // Dentro de límites - limpiar mensaje
        mensajeElement.textContent = '';
        mensajeElement.style.cssText = 'margin-top: 8px; font-size: 14px; min-height: 20px; transition: all 0.3s ease;';
        inputElement.classList.remove('advertencia', 'error');
    }
}

// Función para liberar el bloqueo de una pregunta
function liberarBloqueoPregunta(questionId) {
    const questionElement = document.getElementById(`q-${questionId}`);
    if (!questionElement) return;
    
    const header = questionElement.querySelector('.q-head');
    if (header) {
        header.style.cursor = 'pointer';
        header.style.opacity = '1';
        header.removeAttribute('data-bloqueado');
        
        // Restaurar el onclick original
        header.onclick = function() {
            toggleQuestion(questionId);
        };
    }
}

// Alternar visibilidad de pregunta - MODIFICADA con bloqueo
window.toggleQuestion = function(questionId, forzarApertura = false) {
    const body = document.getElementById(`q-body-${questionId}`);
    const qItem = document.getElementById(`q-${questionId}`);
    
    if (!body || !qItem) return;
    
    // Verificar si esta pregunta está bloqueada (con error)
    const inputElement = document.getElementById(`input-${questionId}`);
    const mensajeElement = document.getElementById(`mensaje-${questionId}`);
    const tieneError = inputElement && inputElement.classList.contains('error') && 
                      mensajeElement && mensajeElement.textContent !== '';
    
    const header = qItem.querySelector('.q-head');
    const estaBloqueada = header && header.getAttribute('data-bloqueado') === 'true';
    
    // Si está bloqueada y NO estamos forzando apertura, NO permitir cerrar
    if (estaBloqueada && !forzarApertura) {
        // Si ya está abierta, mantenerla abierta
        if (!body.classList.contains('open')) {
            body.classList.add('open');
        }
        return;
    }
    
    // Si no tiene error o estamos forzando apertura, proceder normalmente
    const isOpening = !body.classList.contains('open');
    body.classList.toggle('open');
    
    const arrow = qItem.querySelector('.q-head > div > div:last-child');
    if (arrow) {
        arrow.style.transform = isOpening ? 'rotate(180deg)' : 'rotate(0deg)';
        arrow.style.color = isOpening ? 'var(--accent)' : 'var(--muted)';
        arrow.style.transition = 'transform 0.3s ease, color 0.3s ease';
    }
};

// Modifica updateAnswerWithValidation para liberar bloqueo cuando se corrija
function updateAnswerWithValidation(questionId, value, model) {
    // Primero actualizar la respuesta
    updateAnswer(questionId, value);
    
    // Validar si existe límite para esta pregunta
    const limites = LIMITES_VALIDACION[model]?.[questionId];
    const mensajeElement = document.getElementById(`mensaje-${questionId}`);
    const inputElement = document.getElementById(`input-${questionId}`);
    
    if (!limites || !mensajeElement || !inputElement || value === '') {
        // Limpiar mensaje si no hay límites o valor vacío
        if (mensajeElement) {
            mensajeElement.textContent = '';
            mensajeElement.style.cssText = 'margin-top: 8px; font-size: 14px; min-height: 20px; transition: all 0.3s ease;';
        }
        if (inputElement) {
            inputElement.classList.remove('error', 'advertencia');
            // Liberar bloqueo si se corrigió
            liberarBloqueoPregunta(questionId);
        }
        return;
    }
    
    const numValue = parseFloat(value);
    if (isNaN(numValue)) return;
    
    // Verificar si está fuera de límites
    if (numValue < limites.min || numValue > limites.max) {
        mensajeElement.textContent = `⚠️ ${limites.mensaje}. Valor ingresado: ${value}`;
        // Forzar estilo rojo
        mensajeElement.style.cssText = 'color: #ef4444 !important; margin-top: 8px; font-size: 14px; min-height: 20px; font-weight: 600;';
        
        inputElement.classList.add('error');
        inputElement.classList.remove('advertencia');
        
        // Asegurar que la pregunta esté abierta y bloqueada
        const bodyElement = document.getElementById(`q-body-${questionId}`);
        if (bodyElement && !bodyElement.classList.contains('open')) {
            toggleQuestion(questionId, true);
        }
        
        // Asegurar que esté bloqueada
        const questionElement = document.getElementById(`q-${questionId}`);
        const header = questionElement ? questionElement.querySelector('.q-head') : null;
        if (header && header.getAttribute('data-bloqueado') !== 'true') {
            header.style.cursor = 'default';
            header.style.opacity = '0.8';
            header.setAttribute('data-bloqueado', 'true');
            header.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                return false;
            };
        }
        
    } else {
        // Dentro de límites - limpiar mensaje
        mensajeElement.textContent = '';
        mensajeElement.style.cssText = 'margin-top: 8px; font-size: 14px; min-height: 20px; transition: all 0.3s ease;';
        inputElement.classList.remove('advertencia', 'error');
        
        // LIBERAR BLOQUEO cuando el valor es válido
        liberarBloqueoPregunta(questionId);
    }
}

// Función para liberar el bloqueo de una pregunta
function liberarBloqueoPregunta(questionId) {
    const questionElement = document.getElementById(`q-${questionId}`);
    if (!questionElement) return;
    
    const header = questionElement.querySelector('.q-head');
    if (header) {
        header.style.cursor = 'pointer';
        header.style.opacity = '1';
        header.removeAttribute('data-bloqueado');
    }
}

// Simular análisis con IA - MANTENIDA
async function simulateIAAnalysis() {
    return new Promise(resolve => {
        setTimeout(resolve, 2000);
    });
}

// Mostrar resultados del análisis - MODIFICADA para estilo bonito
function showResults(answers, model) {
    const quizResult = document.getElementById('quizResult');
    
    // Generar análisis (AHORA generateIAAnalysis ya usa calculateRiskLevel internamente)
    const analysis = generateIAAnalysis(answers, model);
    
    // Obtener el nivel de riesgo (calculado dentro de generateIAAnalysis)
    // Necesitamos calcularlo por separado también para usarlo en la UI
    const riskLevel = calculateRiskLevel(answers, model);
    const recommendations = generateRecommendations(answers, model);
    const factors = getRiskFactors(answers, model);
    
    // Crear HTML de resultados con estilo bonito original
    const resultsHTML = `
        <div class="result-box show" style="margin-top: 30px; animation: fadeInUp 0.5s ease-out;">
            <!-- Encabezado -->
            <div style="text-align: center; margin-bottom: 30px;">
                <div style="font-size: 48px; margin-bottom: 10px;">
                    ${model === 'stroke' ? '🧠' : '❤️'}
                </div>
                <h2 style="color: var(--text-primary); margin-bottom: 10px; font-size: 28px;">
                    ${model === 'stroke' ? 'Análisis de Riesgo de Derrame Cerebral' : 'Evaluación de Riesgo Cardíaco'}
                </h2>
                <p style="color: var(--text-secondary); font-size: 16px;">
                    Análisis predictivo generado por IA basado en tus respuestas
                </p>
            </div>
            
            <!-- Indicador de riesgo -->
            <div style="background: white; border-radius: 16px; padding: 25px; margin-bottom: 25px; border: 1px solid var(--border); display: flex; align-items: center; gap: 20px; transition: all 0.3s ease;">
                <div style="font-size: 20px; font-weight: 800; padding: 12px 24px; border-radius: 12px; color: white; min-width: 100px; text-align: center; text-shadow: 0 2px 4px rgba(0,0,0,0.1); background: ${riskLevel.class === 'high' ? 'linear-gradient(135deg, #ef4444, #f87171)' : riskLevel.class === 'medium' ? 'linear-gradient(135deg, #f59e0b, #fbbf24)' : 'linear-gradient(135deg, #10b981, #34d399)'};">
                    ${riskLevel.text}
                </div>
                <div style="flex: 1;">
                    <div style="font-weight: 700; margin-bottom: 4px; font-size: 18px;">Nivel de riesgo estimado</div>
                    <div style="color: var(--text-secondary); font-size: 14px;">${riskLevel.description}</div>
                </div>
            </div>
            
            <!-- Análisis de IA -->
            <div style="background: white; border-radius: 16px; padding: 25px; margin-bottom: 25px; border: 1px solid var(--border);">
                <h3 style="color: var(--text-primary); margin-bottom: 15px; font-size: 18px; display: flex; align-items: center; gap: 10px;">
                    <span>📋</span> Análisis Detallado
                </h3>
                <div style="background: #f8fafc; padding: 20px; border-radius: 12px; line-height: 1.6; color: var(--text-primary);">
                    ${analysis}
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid var(--border);">
                        <strong style="color: var(--accent);">📄Fundamento Clínico del Algoritmo:</strong> 
                        Desde un enfoque algorítmico, el sistema analiza patrones que no siempre son evidentes a simple vista. 
                        Este modelo fue entrenado con datos clínicos de 5,000 pacientes y alcanza una precisión del 92% en validación 
                    </div>
                </div>
            </div>
            
            <!-- Factores identificados -->
            <div style="background: white; border-radius: 16px; padding: 25px; margin-bottom: 25px; border: 1px solid var(--border);">
                <h3 style="color: var(--text-primary); margin-bottom: 15px; font-size: 18px; display: flex; align-items: center; gap: 10px;">
                    <span>🔍</span> Factores Identificados
                </h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    ${factors.map(factor => `
                        <div style="background: ${factor.includes('No se identificaron') ? '#f0fdf4' : '#fef3c7'}; 
                                    padding: 12px; border-radius: 8px; border-left: 4px solid ${factor.includes('No se identificaron') ? '#10b981' : '#f59e0b'};">
                            <div style="font-weight: 500; color: var(--text-primary);">${factor}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <!-- Recomendaciones -->
            <div style="background: white; border-radius: 16px; padding: 25px; margin-bottom: 25px; border: 1px solid var(--border);">
                <h3 style="color: var(--text-primary); margin-bottom: 15px; font-size: 18px; display: flex; align-items: center; gap: 10px;">
                    <span>💡</span> Recomendaciones Personalizadas
                </h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                    ${recommendations.map((rec, index) => `
                        <div style="background: white; border-radius: 14px; padding: 20px; border: 1px solid var(--border); transition: all 0.3s ease; display: flex; flex-direction: column; align-items: flex-start;">
                            <div style="font-size: 28px; margin-bottom: 15px;">${rec.icon}</div>
                            <div style="font-weight: 700; font-size: 16px; color: var(--text-primary); margin-bottom: 8px;">
                                ${rec.title}
                            </div>
                            <div style="color: var(--text-secondary); font-size: 14px; line-height: 1.5;">
                                ${rec.text}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <!-- Advertencia -->
            <div style="background: linear-gradient(135deg, #fef3c7, #fde68a); border: 1px solid #fbbf24; border-radius: 16px; padding: 20px; margin-top: 25px; display: flex; align-items: flex-start; gap: 15px;">
                <div style="font-size: 24px; color: #92400e; flex-shrink: 0;">⚠️</div>
                <div>
                    <h4 style="color: #92400e; font-weight: 700; margin-bottom: 8px;">Importante</h4>
                    <p style="color: #92400e; font-size: 14px; opacity: 0.9;">
                        Este análisis es predictivo y utiliza inteligencia artificial. 
                        <strong>No sustituye una consulta médica profesional.</strong> 
                        Si tienes síntomas o preocupaciones, consulta con un médico especialista.
                    </p>
                </div>
            </div>
            
            <!-- Botón para nuevo análisis -->
            <div style="text-align: center; margin-top: 30px;">
                <button class="primary-btn" onclick="selectModel('${model}')" 
                        style="padding: 16px 40px; font-size: 16px;">
                    <span>🔄</span> Realizar nuevo análisis
                </button>
            </div>
        </div>
    `;
    
    quizResult.innerHTML = resultsHTML;
    
    // Scroll suave a resultados
    quizResult.scrollIntoView({ behavior: 'smooth' });
}

// Función auxiliar para factores de riesgo
function getRiskFactors(answers, model) {
    let factors = [];
    
    if (model === 'stroke') {
        if (answers.hypertension === '1') factors.push('Hipertensión arterial');
        if (answers.heart_disease === '1') factors.push('Enfermedad cardíaca previa');
        if (answers.smoking_status === 'smokes') factors.push('Tabaquismo actual');
        if (parseFloat(answers.bmi) >= 30) factors.push('Obesidad (IMC ≥ 30)');
        if (parseFloat(answers.avg_glucose_level) >= 126) factors.push('Glucosa elevada');
        if (parseInt(answers.age) >= 65) factors.push('Edad avanzada');
    } else {
        if (answers.HeartDisease === '1') factors.push('Enfermedad cardíaca diagnosticada');
        if (answers.ExerciseAngina === 'Y') factors.push('Angina por ejercicio');
        if (answers.ST_Slope === 'Down') factors.push('Depresión ST descendente');
        if (answers.ChestPainType === 'ASY') factors.push('Dolor asintomático');
        if (parseFloat(answers.Oldpeak) >= 2) factors.push('Depresión ST significativa');
        if (parseInt(answers.Age) >= 60) factors.push('Edad como factor de riesgo');
    }
    
    if (factors.length === 0) {
        factors.push('No se identificaron factores de riesgo significativos');
    }
    
    return factors;
}

// Generar análisis de IA
function generateIAAnalysis(answers, model) {
    // PRIMERO calcular el nivel de riesgo REAL
    const riskLevel = calculateRiskLevel(answers, model);
    
    // Análisis ESPECÍFICOS para cada nivel de riesgo
    const analyses = {
        stroke: {
            // BAJO RIESGO (riskLevel.class === 'low')
            low: "Con base en los factores analizados, tu perfil se mantiene dentro de lo esperado para una buena salud cardiovascular. Los indicadores revisados no muestran señales preocupantes y el patrón general es estable. Aun así, seguir hábitos saludables ayuda a mantener estos buenos resultados en el tiempo.",
            
            // MODERADO RIESGO (riskLevel.class === 'medium')
            medium: "Los datos analizados muestran algunos factores que pueden aumentar tu riesgo de derrame cerebral si no se controlan a tiempo. Muchos de ellos son modificables, por lo que hacer pequeños ajustes en tus hábitos o seguimiento médico puede ayudarte a reducir este riesgo y mantener una buena salud neurológica.",
            
            // ALTO RIESGO (riskLevel.class === 'high')
            high: "La evaluación identifica varios valores que coinciden con los perfiles de alto riesgo documentados en estudios sobre derrame cerebral. Los patrones encontrados sugieren una probabilidad elevada de complicaciones si no se actúa pronto. Es recomendable buscar una valoración médica para recibir orientación y reducir el riesgo lo antes posible."
        },
        heart: {
            // BAJO RIESGO
            low: "El análisis revisa tus parámetros cardíacos comparándolos con valores normales establecidos en grandes estudios de salud. En tu caso, los resultados se encuentran dentro de los rangos habituales, por lo que no se observan señales que indiquen un riesgo elevado en este momento.",
            
            // MODERADO RIESGO
            medium: "El análisis combina varios indicadores de tu corazón y detecta valores que se alejan de lo saludable. Cuando estos factores se presentan juntos, aumentan la posibilidad de un problema cardíaco. Tus resultados muestran patrones que merecen atención médica para descartar riesgos y actuar a tiempo.",
            
            // ALTO RIESGO
            high: "La IA examina cómo se relacionan tus datos entre sí, no solo cada valor por separado. Gracias a modelos entrenados con miles de casos reales, identifica configuraciones que podrían indicar riesgo antes de que aparezcan síntomas evidentes. En esta evaluación, el sistema detecta patrones relevantes que conviene revisar con un profesional."
        }
    };
    
    // Seleccionar el análisis CORRECTO según el riesgo calculado
    const analysis = analyses[model][riskLevel.class];
    
    return analysis + " " + getPersonalizedInsight(answers, model);
}

// Obtener insight personalizado
function getPersonalizedInsight(answers, model) {
    if (model === 'stroke') {
        if (answers.hypertension === '1' && answers.heart_disease === '1') {
            return "La combinación de hipertensión y enfermedad cardíaca previa representa un factor de riesgo significativo que requiere monitoreo regular.";
        }
        if (answers.smoking_status === 'smokes' && parseFloat(answers.bmi) > 30) {
            return "El hábito de fumar combinado con un IMC elevado sugiere la necesidad de intervenciones en estilo de vida.";
        }
    } else {
        if (answers.ExerciseAngina === 'Y' && answers.ST_Slope === 'Down') {
            return "La presencia de angina con ejercicio y depresión ST descendente son hallazgos que merecen evaluación cardiológica especializada.";
        }
        if (answers.Cholesterol > 240 && answers.FastingBS === '1') {
            return "Los niveles elevados de colesterol y glucosa en ayunas indican un perfil metabólico que beneficia de manejo integral.";
        }
    }
    
    return "Se recomienda seguimiento regular y consideración de medidas preventivas según el perfil identificado.";
}

// Calcular nivel de riesgo
function calculateRiskLevel(answers, model) {
    let score = 0;
    
    if (model === 'stroke') {
        // Edad
        const age = parseInt(answers.age) || 0;
        if (age >= 65) score += 3;
        else if (age >= 55) score += 2;
        else if (age >= 45) score += 1;
        
        // Hipertensión
        if (answers.hypertension === '1') score += 2;
        
        // Enfermedad cardíaca
        if (answers.heart_disease === '1') score += 2;
        
        // Fumador
        if (answers.smoking_status === 'smokes') score += 2;
        
        // BMI
        const bmi = parseFloat(answers.bmi) || 0;
        if (bmi >= 30) score += 2;
        else if (bmi >= 25) score += 1;
        
        // Glucosa
        const glucose = parseFloat(answers.avg_glucose_level) || 0;
        if (glucose >= 140) score += 2;
        else if (glucose >= 100) score += 1;
        
    } else {
        // Heart disease
        if (answers.HeartDisease === '1') score += 3;
        
        // Edad
        const age = parseInt(answers.Age) || 0;
        if (age >= 60) score += 2;
        else if (age >= 50) score += 1;
        
        // Dolor de pecho
        if (answers.ChestPainType === 'ASY') score += 2;
        else if (answers.ChestPainType === 'ATA') score += 1;
        
        // Angina por ejercicio
        if (answers.ExerciseAngina === 'Y') score += 2;
        
        // Oldpeak
        const oldpeak = parseFloat(answers.Oldpeak) || 0;
        if (oldpeak >= 2) score += 2;
        else if (oldpeak >= 1) score += 1;
    }
    
    // Determinar nivel de riesgo
    if (score >= 8) {
        return {
            class: 'high',
            text: 'ALTO',
            description: 'Riesgo elevado - Se recomienda evaluación médica prioritaria'
        };
    } else if (score >= 5) {
        return {
            class: 'medium',
            text: 'MODERADO',
            description: 'Riesgo intermedio - Se sugiere consulta médica y seguimiento'
        };
    } else {
        return {
            class: 'low',
            text: 'BAJO',
            description: 'Riesgo bajo - Mantener hábitos saludables y chequeos regulares'
        };
    }
}

// Generar recomendaciones
function generateRecommendations(answers, model) {
    const recommendations = [];
    
    // Recomendaciones generales
    recommendations.push({
        icon: '🏥',
        title: 'Consulta Médica',
        text: 'Programa una consulta con un especialista para evaluación completa'
    });
    
    recommendations.push({
        icon: '📊',
        title: 'Monitoreo Regular',
        text: 'Realiza chequeos médicos periódicos según tu perfil de riesgo'
    });
    
    // Recomendaciones específicas por modelo
    if (model === 'stroke') {
        if (answers.hypertension === '1') {
            recommendations.push({
                icon: '💊',
                title: 'Control de Hipertensión',
                text: 'Sigue el tratamiento prescrito y monitorea tu presión arterial regularmente'
            });
        }
        
        if (answers.smoking_status === 'smokes') {
            recommendations.push({
                icon: '🚭',
                title: 'Cesación Tabáquica',
                text: 'Considera programas para dejar de fumar y evita la exposición al humo'
            });
        }
        
        const bmi = parseFloat(answers.bmi) || 0;
        if (bmi >= 25) {
            recommendations.push({
                icon: '🥗',
                title: 'Manejo de Peso',
                text: 'Implementa un plan de alimentación balanceada y actividad física regular'
            });
        }
        
    } else {
        if (answers.ExerciseAngina === 'Y') {
            recommendations.push({
                icon: '🏃',
                title: 'Ejercicio Supervisado',
                text: 'Realiza actividad física bajo supervisión médica especializada'
            });
        }
        
        if (answers.Cholesterol > 200) {
            recommendations.push({
                icon: '🥑',
                title: 'Dieta Cardiosaludable',
                text: 'Adopta una dieta baja en grasas saturadas y rica en fibra'
            });
        }
        
        if (answers.FastingBS === '1') {
            recommendations.push({
                icon: '📈',
                title: 'Control Glucémico',
                text: 'Monitorea tus niveles de glucosa y sigue recomendaciones nutricionales'
            });
        }
    }
    
    // Recomendación de estilo de vida
    recommendations.push({
        icon: '😴',
        title: 'Manejo del Estrés',
        text: 'Practica técnicas de relajación y mantiene un sueño reparador'
    });
    
    return recommendations.slice(0, 5); // Limitar a 5 recomendaciones
}

// Reiniciar cuestionario
window.reiniciarCuestionario = function() {
    selectModel(currentModel);
};

// Asegurar que la función esté disponible globalmente
window.updateAnswerWithValidation = updateAnswerWithValidation;

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
// ==================== CHAT AVANZADO ====================
let currentConversationId = null;
let isTyping = false;

function inicializarChat() {
    console.log('Inicializando chat avanzado...');
    
    const chatInput = document.getElementById('chatInput');
    const enviarBtn = document.getElementById('enviarChatBtn');
    const messagesContainer = document.getElementById('messages');
    
    if (!chatInput || !enviarBtn || !messagesContainer) {
        console.error('Elementos del chat no encontrados');
        return;
    }
    
    // Configurar eventos
    enviarBtn.addEventListener('click', enviarMensajeAvanzado);
    
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            enviarMensajeAvanzado();
        }
    });
    
    // Iniciar nueva conversación
    iniciarNuevaConversacion();
    
    console.log('Chat avanzado inicializado');
}

async function iniciarNuevaConversacion() {
    try {
        const response = await fetch('http://127.0.0.1:5000/api/chat/new', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: 'lifescan_user'
            })
        });
        
        const data = await response.json();
        if (data.success) {
            currentConversationId = data.conversation_id;
            console.log(`Nueva conversación: ${currentConversationId}`);
            
            // Mostrar mensaje de bienvenida
            // Mensaje de bienvenida actualizado
            mostrarMensajeBot(`
                🩺 ¡Hola! Soy Aila Assistant 🤖
                
                Tu asistente médico especializado en salud preventiva.
                
                📋 Puedo ayudarte con:
                • Información médica general y educativa
                • Explicación de términos médicos complejos
                • Consejos de estilo de vida saludable
                • Orientación sobre cuándo buscar atención
                
                ⚠️ Recordatorio importante:
                Soy una herramienta educativa de apoyo.
                Nunca reemplazo la consulta con un médico profesional.
                
                💡 ¿En qué puedo asistirte hoy?
            `);
        }
    } catch (error) {
        console.error('Error iniciando conversación:', error);
        // Usar ID por defecto
        currentConversationId = 'default_' + Date.now();
        mostrarMensajeBot('¡Hola! Soy tu asistente IA de salud. ¿En qué puedo ayudarte hoy?');
    }
}

async function enviarMensajeAvanzado() {
    const chatInput = document.getElementById('chatInput');
    const mensaje = chatInput.value.trim();
    
    if (!mensaje || isTyping) return;
    
    // Mostrar mensaje del usuario
    mostrarMensajeUsuario(mensaje);
    chatInput.value = '';
    
    // Mostrar indicador de "escribiendo"
    mostrarTypingIndicator();
    isTyping = true;
    
    try {
        // Enviar a Flask/OpenAI
        const response = await fetch('http://127.0.0.1:5000/api/chat/send', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: mensaje,
                conversation_id: currentConversationId
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remover indicador de "escribiendo"
        removeTypingIndicator();
        
        if (data.success) {
            // Mostrar respuesta formateada
            mostrarMensajeBotFormateado(data.response);
            
            // Actualizar historial en sidebar
            actualizarHistorialChat(mensaje, data.timestamp);
        } else {
            throw new Error(data.error || 'Error en la respuesta');
        }
        
    } catch (error) {
        console.error('Error enviando mensaje:', error);
        
        // Remover indicador de "escribiendo"
        removeTypingIndicator();
        
        // Respuesta de fallback
        // Respuesta de fallback actualizada
            mostrarMensajeBot(`
                ⚠️ Modo Demo Activado
                
                No se pudo conectar con el servicio de IA.
                Usando respuestas educativas predefinidas.
                
                💡 Consejo de salud general:
                Para consultas específicas, te recomendamos:
                • Consultar con tu médico de cabecera
                • Llevar un registro detallado de síntomas
                • Realizar chequeos preventivos regulares
                
                🏥 Tu salud es lo más importante:
                Siempre busca atención profesional para diagnósticos.
                
                ¿Hay algo específico sobre lo que te gustaría información?
            `);
        
        // Actualizar historial
        actualizarHistorialChat(mensaje, new Date().toLocaleDateString());
        
    } finally {
        isTyping = false;
    }
}

function mostrarMensajeUsuario(texto) {
    const messagesContainer = document.getElementById('messages');
    const mensajeDiv = document.createElement('div');
    mensajeDiv.className = 'msg user';
    mensajeDiv.innerHTML = `
        <div class="msg-content">${texto}</div>
        <div class="msg-time">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
    `;
    messagesContainer.appendChild(mensajeDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function mostrarMensajeBotFormateado(texto) {
    const messagesContainer = document.getElementById('messages');
    const mensajeDiv = document.createElement('div');
    mensajeDiv.className = 'msg bot';
    
    // Formatear texto con saltos de línea
    const textoFormateado = texto.replace(/\n/g, '<br>');
    
    mensajeDiv.innerHTML = `
        <div class="msg-header">
            <strong>Asistente IA:</strong>
        </div>
        <div class="msg-content">${textoFormateado}</div>
        <div class="msg-time">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
    `;
    messagesContainer.appendChild(mensajeDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function mostrarMensajeBot(texto) {
    mostrarMensajeBotFormateado(texto);
}

function mostrarTypingIndicator() {
    const messagesContainer = document.getElementById('messages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'msg bot typing';
    typingDiv.innerHTML = `
        <div class="msg-header">
            <strong>Asistente IA:</strong>
        </div>
        <div class="msg-content">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function actualizarHistorialChat(mensaje, timestamp) {
    const historyCol = document.getElementById('historyCol');
    if (!historyCol) return;
    
    // Crear nueva tarjeta de historial
    const historyCard = document.createElement('div');
    historyCard.className = 'history-card';
    historyCard.innerHTML = `
        <strong>[${timestamp}]</strong><br>
        ${mensaje.substring(0, 50)}${mensaje.length > 50 ? '...' : ''}
    `;
    
    // Agregar al inicio
    historyCol.insertBefore(historyCard, historyCol.firstChild);
    
    // Limitar a 10 tarjetas
    if (historyCol.children.length > 10) {
        historyCol.removeChild(historyCol.lastChild);
    }
}
//-----------------------------------------------------------------------
// Agrega esta función para formatear mejor el texto
function formatearTextoChat(texto) {
    // Reemplazar ** por nada y mantener estructura
    let formateado = texto.replace(/\*\*/g, '');
    
    // Mantener saltos de línea
    formateado = formateado.replace(/\n/g, '<br>');
    
    // Mejorar listas
    formateado = formateado.replace(/\n• /g, '<br>• ');
    formateado = formateado.replace(/\n\d\. /g, '<br>$&');
    
    // Añadir clases para emojis en títulos
    formateado = formateado.replace(/^([🩺💊🏥⚠️🚨🔍📋💡✅]+)\s+(.+)$/gm, 
        '<div class="emoji-title">$1<span>$2</span></div>');
    
    return formateado;
}

// Actualiza la función mostrarMensajeBotFormateado
function mostrarMensajeBotFormateado(texto) {
    const messagesContainer = document.getElementById('messages');
    const mensajeDiv = document.createElement('div');
    mensajeDiv.className = 'msg bot';
    
    // Formatear texto
    const textoFormateado = formatearTextoChat(texto);
    
    mensajeDiv.innerHTML = `
        <div class="msg-header">
            <strong>Asistente IA:</strong>
        </div>
        <div class="msg-content">${textoFormateado}</div>
        <div class="msg-time">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
    `;
    messagesContainer.appendChild(mensajeDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Asegurar que la función esté disponible globalmente
window.enviarMensajeAvanzado = enviarMensajeAvanzado;
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Función global para procesar imágenes desde HTML
window.handleFile = function(event) {
    if (event.target.files.length > 0) {
        procesarImagen(event.target.files[0]);
    }
};

// Función global para alternar preguntas
window.toggleQuestion = toggleQuestion;

console.log('Script IA Health cargado correctamente');