
# Импорт библиотек
import numpy as np                   # работа с массивами и математическими операциями
import scipy.io.wavfile as wf        # сохранение аудио в WAV‑файл
import os                            # работа с файловой системой (создание директорий)
import configparser

class Payload:
    """Класс Payload
    Преобразует текст в битовый массив (ITA2/МТК-2) с автоматическим переключением режимов"""
    ITA2 = {
        # латинские буквы в стандартном коде ITA2
        'A': [1, 1, 0, 0, 0], 'B': [1, 0, 0, 1, 1], 'C': [0, 1, 1, 1, 0],
        'D': [1, 0, 0, 1, 0], 'E': [1, 0, 0, 0, 0], 'F': [1, 0, 1, 1, 0],
        'G': [0, 1, 0, 1, 1], 'H': [0, 0, 1, 0, 1], 'I': [0, 1, 1, 0, 0],
        'J': [1, 1, 0, 1, 0], 'K': [1, 1, 1, 1, 0], 'L': [0, 1, 0, 0, 1],
        'M': [0, 0, 1, 1, 1], 'N': [0, 0, 1, 1, 0], 'O': [0, 0, 0, 1, 1],
        'P': [0, 1, 1, 0, 1], 'Q': [1, 1, 1, 0, 1], 'R': [0, 1, 0, 1, 0],
        'S': [1, 0, 1, 0, 0], 'T': [0, 0, 0, 0, 1], 'U': [1, 1, 1, 0, 0],
        'V': [0, 1, 1, 1, 1], 'W': [1, 1, 0, 0, 1], 'X': [1, 0, 1, 1, 1],
        'Y': [1, 0, 1, 0, 1], 'Z': [1, 0, 0, 0, 1],
        # цифры/знаки в стандартном коде МТК-2
        '0': [0, 1, 1, 0, 1], '1': [1, 1, 1, 0, 1], '2': [1, 1, 0, 0, 1],
        '3': [1, 0, 0, 0, 0], '4': [0, 1, 0, 1, 0], '5': [0, 0, 0, 0, 1],
        '6': [1, 0, 1, 0, 1], '7': [1, 1, 1, 0, 0], '8': [0, 1, 1, 0, 0],
        '9': [0, 0, 0, 1, 1], '-': [1, 1, 0, 0, 0], '+': [1, 0, 0, 0, 1],
        '?': [1, 0, 0, 1, 1], ':': [0, 1, 1, 1, 0], '(': [1, 1, 1, 1, 0],
        ')': [0, 1, 0, 0, 1], '.': [0, 0, 1, 1, 1], ',': [0, 0, 1, 1, 0],
        '/': [0, 1, 1, 1, 1], ' ': [0, 0, 1, 0, 0],
        'Ш': [0, 1, 0, 1, 1], 'Щ': [0, 0, 1, 0, 1], 'Э': [1, 0, 1, 1, 0],
        'Ю': [1, 1, 0, 1, 0], 'Ч': [0, 1, 0, 1, 0],
        # Русские буквы в стандартном коде МТК-2
        'А': [1, 1, 0, 0, 0], 'Б': [1, 0, 0, 1, 1], 'В': [1, 1, 0, 0, 1],
        'Г': [0, 1, 0, 1, 1], 'Д': [1, 0, 0, 1, 0], 'Е': [1, 0, 0, 0, 0],
        'Ж': [0, 1, 1, 1, 1], 'З': [1, 0, 0, 0, 1], 'И': [0, 1, 1, 0, 0],
        'Й': [1, 1, 0, 1, 0], 'К': [1, 1, 1, 1, 0], 'Л': [0, 1, 0, 0, 1],
        'М': [0, 0, 1, 1, 1], 'Н': [0, 0, 1, 1, 0], 'О': [0, 0, 0, 1, 1],
        'П': [0, 1, 1, 0, 1], 'Р': [0, 1, 0, 1, 0], 'С': [1, 0, 1, 0, 0],
        'Т': [0, 0, 0, 0, 1], 'У': [1, 1, 1, 0, 0], 'Ф': [1, 0, 1, 1, 0],
        'Х': [0, 0, 1, 0, 1], 'Ц': [0, 1, 1, 1, 0], 'Ъ': [1, 0, 1, 1, 1],
        'Ы': [1, 0, 1, 0, 1], 'Ь': [1, 0, 1, 1, 1], 
        'Я': [1, 1, 1, 0, 1], 'Ё': [1, 0, 0, 0, 0],
        'RUS' : [0, 0, 0, 0, 0],  # переключение на русские буквы
        'FIGS': [1, 1, 0, 1, 1],  # переключение на цифры/спецсимволы
        'LAT':  [1, 1, 1, 1, 1],  # переключение на латинские буквы
        '\r':   [0, 0, 0, 1, 0],  # CR (возврат каретки)
        '\n':   [0, 1, 0, 0, 0]   # LF (перевод строки)
    }
    
    # инициализация
    def __init__(self, text):
        """Метод __init__ класса Payload
		Конструктор класса, который:
		- Принимает входной текст,
		- Подготавливает внутренние атрибуты объекта,
		- Запускает процесс кодирования текста в битовый массив по стандарту ITA2.
		self — ссылка на создаваемый экземпляр класса,
		text — входной текст (строка), который нужно закодировать.
		"""
        self.rawData = str(text).upper()
                                   # Преобразует входной параметр text в строку (если он ещё не строка).
                                   # Приводит все символы к верхнему регистру с помощью .upper() (ITA2 работает только с заглавными буквами).
                                   # Сохраняет результат в атрибут rawData.
        self.bitArray = []         # Создаёт пустой список для хранения итогового битового представления сообщения.
                                   # В дальнейшем сюда будут добавляться биты (0, 1 и 0.5) по правилам ITA2.
        self.bits_length = 0       # Инициализирует счётчик длины битового массива.
                                   # После кодирования будет обновлён до реальной длины bitArray.
        self.current_mode = 'LAT'  # Устанавливает начальный режим кодирования: 'LAT' (латинские буквы).
        self._encode_to_ita2()     # Вызывает приватный метод _encode_to_ita2(), который:
                                   # - Добавляет служебные символы (CR+LF) в начало и конец.
                                   # - Анализирует каждый символ текста.
                                   # - При необходимости переключает режим (например, с 'LAT' на 'RUS').
                                   # - Кодирует символы в биты по таблице ITA2.
                                   # - Формирует итоговый bitArray с старт‑, стоп‑ и полустоп‑битами.

    def _encode_to_ita2(self):
	    """Метод _encode_to_ita2 класса Payload
	    преобразует исходный текст self.rawData в последовательность битов self.bitArray с учётом:
	    - служебных символов (CR/LF);
	    - переключения между режимами (LAT/RUS/FIGS);
	    - правил ITA2 (старт‑, стоп‑ и полустоп‑биты)."""
	    # 1. Добавление служебных символов CR + LF в начало
	    self._add_char_to_bitarray('LAT')               # добавляем в битовый массив (bitArray) код режима LAT
	    self.current_mode = 'LAT'                       # устанавливаем текущий режим LAT
	    self._add_char_to_bitarray('\r')                # добавляем в битовый массив (bitArray) код символа CR
	    self._add_char_to_bitarray('\n')                # добавляем в битовый массив (bitArray) код символа LF
	
	    # 2. Кодируем основной текст
	    for char in self.rawData:       # Цикл по символам текста: для каждого символа
	        if char not in self.ITA2:   #   Проверка поддержки: если символ отсутствует в таблице ITA2,
	            continue                #                       пропускаем неподдерживаемые символы
	
	        target_mode = self._get_char_mode(char)     # Вызываем метод _get_char_mode(char), который по символу char определяет, в каком режиме он должен кодироваться
	                                                    # Например, если символ в маасиве 'ABC...XYZ' → возвращает 'LAT'.
	        if target_mode != self.current_mode:        # Проверяет, совпадает ли требуемый режим (target_mode) с текущим режимом объекта (self.current_mode).
	                                                    # Если режимы различаются, то
	            self._add_char_to_bitarray(target_mode) # Вызываем метод _add_char_to_bitarray с аргументом target_mode, который
	                                                    # добавляет в битовый массив (bitArray) код переключения режима
	                                                    # Например, при переключении на 'RUS' в bitArray добавится:
	                                                    # [0, 0, 0, 0, 0, 0, 1, 0.5] (старт + код RUS + стоп + полустоп).
	            self.current_mode = target_mode         # Обновляем текущий режим объекта на новый (target_mode)
	                                                    # чтобы не отправлять коды режима до следующего переключения.
	        self._add_char_to_bitarray(char)            # Вызываем метод _add_char_to_bitarray с аргументом char, который
	                                                    # добавляет в битовый массив (bitArray) код символа
	                                                    # Например, для символа 'А' в режиме 'RUS' добавится:
	                                                    # [0, 1, 1, 0, 0, 0, 1, 0.5] (старт + код RUS + стоп + полустоп).
	    # 3. Добавление служебных символов CR + LF в конец
	    self._add_char_to_bitarray('\r')                # добавляем в битовый массив (bitArray) код символа CR
	    self._add_char_to_bitarray('\n')                # добавляем в битовый массив (bitArray) код символа LF
	    self.bits_length = len(self.bitArray)           # Обновляем атрибут bits_length — длину битового массива после завершения кодирования.


    def _get_char_mode(self, char):
        """Метод _get_char_mode класса Payload
	    Определяет режим, к которому относится символ (LAT, FIGS или RUS).
	    self — ссылка на создаваемый экземпляр класса,
	    char - символ, по которому будет определён режим.
	    Метод ожидает заглавные буквы.
	    """
        if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':        # если символ относится к латинским буквам
            return 'LAT'                                # вернуть значение 'LAT'
        elif char in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЪЫЬЯ':    # если символ относится к русским буквам (за исключением Ч,Ш,Щ,Э,Ю)
            return 'RUS'                                # вернуть значение 'RUS'
        else:                                           # во всех остальных случаях (цифры, знаки препинания, пробел, буквы Ч,Ш,Щ,Э,Ю и др.)
            return 'FIGS'                               # вернуть значение 'FIGS'


    def _add_char_to_bitarray(self, char):
        """метод _add_char_to_bitarray класса Payload
        Добавляет код символа в bitArray с старт/стоп‑битами"""
        code = self.ITA2[char]                          # Получает 5‑битный код символа из словаря ITA2.
                                                        # Если символ отсутствует в ITA2, возникнет исключение KeyError
                                                        # поэтому предварительно в _encode_to_ita2 выполняется проверка
        self.bitArray.extend([0] + code + [1, 0.5])     # Формирует полную последовательность: старт, код, стоп, полустоп
                                                        # и добавляет сформированные элементы поодиночке в конец bitArray


class RTTYSignal:
    """Класс RTTYSignal
    осуществляет генерацию звукового FSK‑сигнала (Frequency Shift Keying) по закодированным данным из объекта Payload,
    т.е. преобразует битовый поток (payload.bitArray) в аудиосигнал"""

    def __init__(self, payload, baud, mark_freq, space_freq, amplitude, sample_rate):
        self.payload = payload                          # объект Payload с закодированными битами
        self.baud = baud                                # скорость передачи (по умолчанию 45.45 бод).
        self.mark_freq = mark_freq                      # частота марк  (обычно выше, чем спейс)
        self.space_freq = space_freq                    # частота спейс (обычно ниже, чем марк)
        self.amplitude = amplitude                      # амплитуда сигнала (по умолчанию 10 000)
        self.signal = np.array([])                      # итоговый аудиосигнал (пустой до вызова modulate)
        self.isModulated = False                        # флаг: сигнал сгенерирован (True) или нет.
        self.sample_rate = sample_rate                  # частота дискретизации (44 100 Гц).
        self.mark_tone_duration = 0.1                   # 100 мс (длительность маркерного тона в начале и конце передачи)
        self.end_pause_duration = 0.1                   # 100 мс (длительность паузы в конце передачи)

    def _generate_tone(self, freq, duration, start_phase):
        """Генерирует тон заданной частоты и длительности (на основе синуса)."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False) # t - массив временных отсчётов (через np.linspace)
        # Учитываем начальную фазу
        phase = 2 * np.pi * freq * t + start_phase      # phase — фазовая функция: 2π*freq*t+start_phase
        return self.amplitude * np.sin(phase)           # Возвращает: amplitude*sin(phase)

    def _generate_silence(self, duration):
        """Генерирует тишину (нулевой сигнал) заданной длительности."""
        n_samples = int(self.sample_rate * duration)    # Вычисляет число отсчётов: int(sample_rate * duration)
        return np.zeros(n_samples)                      # Возвращает np.zeros(n_samples)

    def modulate(self):
        """Генерирует FSK‑сигнал без пауз между битами, с 1‑сек паузой в конце."""
        bit_duration = 1.0 / self.baud                  # длительность одного бита в секундах
        total_phase = 0.0                               # Текущая фаза сигнала


        # 1. Маркерный тон ПЕРЕД передачей (начинаем с фазы 0)
        pre_mark = self._generate_tone(self.mark_freq, self.mark_tone_duration, 0.0)

        # 2. Сигнал данных (без пауз между битами)
        data_segments = []                              # список сегментов (только биты)
        for bit in self.payload.bitArray:               # цикл по битам из payload.bitArray
            # Определяем частоту для бита
            if bit == 1 or bit == 0.5:                  # если mark (включая полустоп 1 или 0.5)
                freq = self.mark_freq                   # частота freq = марк
            else:                                       # иначе - space (0)
                freq = self.space_freq                  # частота freq = спейс

            # Определяем длительность бита (учитываем полустоп 0.5)
            if bit == 0.5:                              # если полустоп (0.5)
                bit_dur = 0.5 * bit_duration            # длительность bit_dur = 0.5*bit_duration
            else:                                       # иначе (0 или 1)
                bit_dur = bit_duration                  # длительность bit_dur = bit_duration

            # Генерируем сигнал бита в сегмент
            bit_signal = self._generate_tone(freq, bit_dur, total_phase)  # генерируем сигнал с частотой freq, длительностью bit_dur, фазой total_phase
            data_segments.append(bit_signal)                              # Добавляет сгенерированный звуковой фрагмент (из одного бита/полубита) в список сегментов сигнала
            
            # Обновляем общую фазу для следующего бита
            cycles = freq * bit_dur                     # Количество циклов колебаний за время звучания бита/полубита
            total_phase += 2 * np.pi * cycles           # Переводим количество циклов в фазовый угол (в радианах) 
                                                        # и добавляем этот прирост к текущей фазе total_phase
                                                        # чтобы следующий бит начался с той фазы, на которой закончился предыдущий (плавный переход)
            total_phase %= 2 * np.pi                    # Нормализуем фазу [0, 2π)
                                                        # Операция %= (остаток от деления) «обрубает» лишние полные обороты, оставляя только дробную часть.
        
        # Объединяем все сегменты данных
        data_signal = np.concatenate(data_segments)

        # 3. Маркерный тон ПОСЛЕ передачи
        post_mark = self._generate_tone(self.mark_freq, self.mark_tone_duration, total_phase)

        # 4. Пауза в конце
        end_pause = self._generate_silence(self.end_pause_duration)

        # 5. Итоговый сигнал: pre_mark + data_signal + post_mark + end_pause
        self.signal = np.concatenate([pre_mark, data_signal, post_mark, end_pause])
        self.isModulated = True                         # Устанавливает флаг, что сигнал успешно сгенерирован.


    def play(self):
        """Воспроизводит сигнал через звуковую карту."""
        if not self.isModulated:                        # Если сигнал не сгенерирован (isModulated == False), выводится ошибка и метод завершается
            print("Ошибка: сигнал не сгенерирован.")
            return
        try:                                                              # Попытка воспроизведения
            import sounddevice as sd                                      # Подключаем библиотеку для работы со звуковой картой
            # Нормализуем сигнал до диапазона [-1, 1] для безопасного воспроизведения
            signal_normalized = self.signal / np.max(np.abs(self.signal)) # Делим массив отсчётов на максимальное абсолютное значение в сигнале
                                                                          # что приводит сигнал к диапазону [-1, 1], т.е. предотвращает клиппинг
            print(" ")
            print("Параметры сигнала ", self.sample_rate, "Гц 16 бит")
            print("Скорость передачи ", self.baud, "бод")
            print("Частота МАРК      ", self.mark_freq, " Гц")
            print("Частота СПЕЙС     ", self.space_freq, " Гц")
            print(" ")
            print("Воспроизведение запущено...")
            
            sd.play(signal_normalized, samplerate=self.sample_rate)       # Отправляем нормализованный сигнал на звуковую карту с частотой дискретизации self.sample_rate
            sd.wait()                                                     # Ждём окончания воспроизведения
                                                                          # код не продолжит выполнение, пока не закончится звук.
            print("Воспроизведение завершено.")
        except ModuleNotFoundError:
            print("Ошибка: библиотека sounddevice не установлена. Выполните: pip install sounddevice")
        except Exception as e:
            print(f"Ошибка при воспроизведении: {e}")

    def save_wav(self, filename=None):
        """Сохраняет сигнал в WAV‑файл."""
        if not self.isModulated:                        # Если сигнал не сгенерирован (isModulated == False), выводится ошибка и метод завершается
            print("Ошибка: сигнал не сгенерирован.")
            return
        if filename is None:                            # Если имя файла не передано, 
            filename = self.payload.rawData[:20]        # берётся первые 20 символов исходного текста сообщения
        os.makedirs("./audio", exist_ok=True)           # Создаёт директорию ./audio/, если её нет (exist_ok=True подавляет ошибку при существовании папки)
        filepath = f"./audio/{filename}.wav"            # Полный путь к файлу (например, ./audio/HELLO.wav)
        try:
            # Нормализация сигнала до диапазона int16
            signal_normalized = self.signal / np.max(np.abs(self.signal))   # Нормализация сигнала к диапазону [-1, 1]
            signal_int16 = np.int16(signal_normalized * 32767)              # Умножение на 32767 масштабирует сигнал до максимального значения int16
            wf.write(filepath, self.sample_rate, signal_int16)              # Функция из модуля scipy.io.wavfile для записи в WAV-файл
            print(f"Файл сохранён: {filepath}")
        except Exception as e:
            print(f"Ошибка при сохранении: {e}")
                                                                            # WAV сохраняется в несжатом PCM‑формате (16 бит, моно).


def read_config(config_file):
    """Читает INI‑файл, извлекает текст между START_TEXT_MARKER и END_TEXT_MARKER,
    затем читает остальные параметры, игнорируя блок текста."""
    encodings = ['utf-8-sig', 'windows-1251', 'cp1251']
    text_lines = []
    is_inside_text = False
    stripped_lines = []  # Для повторного использования без маркеров

    # 1. Извлекаем текст между маркерами и сохраняем «очищенные» строки
    for enc in encodings:
        try:
            with open(config_file, 'r', encoding=enc) as f:
                lines = f.readlines()

            text_lines.clear()
            is_inside_text = False
            stripped_lines.clear()

            for line in lines:
                stripped_line = line.strip()

                if stripped_line == 'START_TEXT_MARKER':
                    is_inside_text = True
                    continue

                if stripped_line == 'END_TEXT_MARKER':
                    is_inside_text = False
                    continue  # Пропускаем саму строку с маркером

                if is_inside_text:
                    text_lines.append(line.rstrip('\n'))  # Сохраняем текст
                else:
                    stripped_lines.append(line)  # Сохраняем остальные строки

            if text_lines:  # Если текст найден, выходим из цикла кодировок
                print(f"Файл {config_file} прочитан с кодировкой: {enc}")
                break

        except Exception as e:
            print(f"Ошибка при чтении с кодировкой {enc}: {e}")
            continue
    else:
        raise ValueError(f"Не удалось прочитать файл {config_file} ни в одной из кодировок: {encodings}")

    # 2. Собираем «чистый» INI‑контент без блока текста
    ini_content = ''.join(stripped_lines)

    # 3. Читаем параметры через configparser из очищенного контента
    config = configparser.ConfigParser()
    try:
        config.read_string(ini_content, source=config_file)
    except configparser.ParsingError as e:
        raise ValueError(f"Ошибка парсинга INI после удаления текста: {e}")

    # 4. Извлекаем параметры
    baud = float(config['Signal']['baud'])
    mark_freq = int(config['Signal']['mark_freq'])
    space_freq = int(config['Signal']['space_freq'])
    amplitude = int(config['Signal']['amplitude'])
    sample_rate = int(config['Signal']['sample_rate'])
    filename = config['Output']['filename']

    # 5. Формируем итоговый текст с переносами
    text = '\n'.join(text_lines)

    return text, baud, mark_freq, space_freq, amplitude, sample_rate, filename



if __name__ == "__main__":
    config_file = 'rtty.ini'  # имя конфигурационного файла
    print("Система генерации RTTY-сигнала (радиотелетайп)")
    try:
        # Читаем параметры из INI-файла
        text, baud, mark_freq, space_freq, amplitud, sample_rate, filename = read_config(config_file)

        # Создаём payload и сигнал
        payload = Payload(text)
        rtty = RTTYSignal(
            payload,
            baud=baud,
            mark_freq=mark_freq,
            space_freq=space_freq,
            amplitude=amplitud,
            sample_rate=sample_rate
        )

        rtty.modulate()
        rtty.play()
        rtty.save_wav(filename)

    except FileNotFoundError:
        print(f"Ошибка: файл конфигурации '{config_file}' не найден.")
    except KeyError as e:
        print(f"Ошибка в структуре INI‑файла: отсутствует секция или параметр — {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")

