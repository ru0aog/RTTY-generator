import numpy as np
import scipy.io.wavfile as wf
import os


class Payload:
    """Преобразует текст в битовый массив (ITA2) с автоматическим переключением режимов."""
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
        # цифры/знаки в стандартном коде ITA2
        '0': [0, 1, 1, 0, 1], '1': [1, 1, 1, 0, 1], '2': [1, 1, 0, 0, 1],
        '3': [1, 0, 0, 0, 0], '4': [0, 1, 0, 1, 0], '5': [0, 0, 0, 0, 1],
        '6': [1, 0, 1, 0, 1], '7': [1, 1, 1, 0, 0], '8': [0, 1, 1, 0, 0],
        '9': [0, 0, 0, 1, 1], '-': [1, 1, 0, 0, 0], '+': [1, 0, 0, 0, 1],
        '?': [1, 0, 0, 1, 1], ':': [0, 1, 1, 1, 0], '(': [1, 1, 1, 1, 0],
        ')': [0, 1, 0, 0, 1], '.': [0, 0, 1, 1, 1], ',': [0, 0, 1, 1, 0],
        '/': [0, 1, 1, 1, 1], ' ': [0, 0, 1, 0, 0],
        'Ш': [0, 1, 0, 1, 1], 'Щ': [0, 0, 1, 0, 1], 'Э': [1, 0, 1, 1, 0],
        'Ю': [1, 1, 0, 1, 0], 'Ч': [0, 1, 0, 1, 0],
        # Русские буквы (только в режиме RUS)
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
        'RUS' : [0, 0, 0, 0, 0],  # ПЕРЕКЛЮЧЕНИЕ НА РУССКИЙ РЕЖИМ
        'FIGS': [1, 1, 0, 1, 1],  # переключение на цифры/спецсимволы
        'LAT': [1, 1, 1, 1, 1],   # переключение на латинские буквы
        '\r': [0, 0, 0, 1, 0],    # CR (возврат каретки)
        '\n': [0, 1, 0, 0, 0]     # LF (перевод строки)
    }


    def __init__(self, text):
        self.rawData = str(text).upper()
        self.bitArray = []
        self.bits_length = 0
        self.current_mode = 'LAT'  # начальный режим — буквы
        self._encode_to_ita2()

    
    def _encode_to_ita2(self):
	    """Кодирует текст в биты по ITA2 с автоматическим переключением режимов."""
	    # 1. Добавляем CR + LF в начало
	    self._add_char_to_bitarray('LAT')
	    self._add_char_to_bitarray('\r')
	    self._add_char_to_bitarray('\n')
	
	    # 2. Начальное переключение на LTRS (если ещё не в этом режиме)
	    if self.current_mode != 'LAT':
	        self._add_char_to_bitarray('LAT')
	        self.current_mode = 'LAT'
	
	    # 3. Кодируем основной текст
	    for char in self.rawData:
	        if char not in self.ITA2:
	            continue  # пропускаем неподдерживаемые символы
	
	        target_mode = self._get_char_mode(char)
	        if target_mode != self.current_mode:
	            self._add_char_to_bitarray(target_mode)
	            self.current_mode = target_mode
	        self._add_char_to_bitarray(char)
	
	    # 4. Добавляем CR + LF в конец
	    self._add_char_to_bitarray('\r')
	    self._add_char_to_bitarray('\n')
	    self.bits_length = len(self.bitArray)



    def _get_char_mode(self, char):
        """Определяет, в каком режиме должен быть символ (LTRS, FIGS или RUS)."""
        if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return 'LAT'
        elif char in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЪЫЬЯ':
            return 'RUS'
        else:
            return 'FIGS'  # цифры, знаки препинания, пробел и др.


    def _add_char_to_bitarray(self, char):
        """Добавляет символ в bitArray с старт/стоп‑битами."""
        code = self.ITA2[char]
        self.bitArray.extend([0] + code + [1, 0.5])  # старт, код, стоп, полустоп



class RTTYSignal:
    """Генерирует RTTY‑сигнал (FSK) из текстового сообщения."""

    def __init__(self, payload, baud=45.45, mark_freq=1170, space_freq=1000, amplitude=10000):
        self.payload = payload
        self.baud = baud
        self.mark_freq = mark_freq
        self.space_freq = space_freq
        self.amplitude = amplitude
        self.signal = np.array([])
        self.isModulated = False
        self.sample_rate = 44100  # Гц
        self.mark_tone_duration = 0.1  # 100 мс (маркерный тон)
        self.end_pause_duration = 0.1  # 0.1 секунда паузы в конце

    def _generate_tone(self, freq, duration):
        """Генерирует тон заданной частоты и длительности (на основе синуса)."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        return self.amplitude * np.sin(2 * np.pi * freq * t)

    def _generate_silence(self, duration):
        """Генерирует тишину (нулевой сигнал) заданной длительности."""
        n_samples = int(self.sample_rate * duration)
        return np.zeros(n_samples)

    def modulate(self):
        """Генерирует FSK‑сигнал без пауз между битами, с 1‑сек паузой в конце."""
        bit_duration = 1.0 / self.baud  # длительность одного бита в секундах


        # 1. Маркерный тон ПЕРЕД передачей
        pre_mark = self._generate_tone(self.mark_freq, self.mark_tone_duration)

        # 2. Сигнал данных (без пауз между битами)
        data_segments = []  # список сегментов (только биты)
        for bit in self.payload.bitArray:
            # Определяем частоту для бита
            if bit == 1 or bit == 0.5:  # mark (включая полустоп)
                freq = self.mark_freq
            else:  # space (0)
                freq = self.space_freq
            
            # Длительность бита (учитываем полустоп 0.5)
            if bit == 0.5:
                bit_dur = 0.5 * bit_duration
            else:
                bit_dur = bit_duration
            
            # Генерируем сигнал бита
            bit_signal = self._generate_tone(freq, bit_dur)
            data_segments.append(bit_signal)
        
        
        # Объединяем все сегменты данных
        data_signal = np.concatenate(data_segments)


        # 3. Маркерный тон ПОСЛЕ передачи
        post_mark = self._generate_tone(self.mark_freq, self.mark_tone_duration)

        # 4. Пауза в конце (1 секунда тишины)
        end_pause = self._generate_silence(self.end_pause_duration)

        # 5. Итоговый сигнал: pre_mark + data_signal + post_mark + end_pause
        self.signal = np.concatenate([pre_mark, data_signal, post_mark, end_pause])
        self.isModulated = True


    def play(self):
        """Воспроизводит сигнал через звуковую карту."""
        if not self.isModulated:
            print("Ошибка: сигнал не сгенерирован.")
            return

        try:
            import sounddevice as sd

            # Нормализуем сигнал до диапазона [-1, 1] для безопасного воспроизведения
            signal_normalized = self.signal / np.max(np.abs(self.signal))

            print("Воспроизведение запущено...")
            sd.play(signal_normalized, samplerate=self.sample_rate)
            sd.wait()  # Ждём окончания воспроизведения
            print("Воспроизведение завершено.")
        except ModuleNotFoundError:
            print("Ошибка: библиотека sounddevice не установлена. Выполните: pip install sounddevice")
        except Exception as e:
            print(f"Ошибка при воспроизведении: {e}")

    def save_wav(self, filename=None):
        """Сохраняет сигнал в WAV‑файл."""
        if not self.isModulated:
            print("Ошибка: сигнал не сгенерирован.")
            return
        if filename is None:
            filename = self.payload.rawData[:20]
        os.makedirs("./audio", exist_ok=True)
        filepath = f"./audio/{filename}.wav"
        try:
            # Нормализация сигнала до диапазона int16
            signal_normalized = self.signal / np.max(np.abs(self.signal))
            signal_int16 = np.int16(signal_normalized * 32767)
            wf.write(filepath, self.sample_rate, signal_int16)
            print(f"Файл сохранён: {filepath} (частота дискретизации: {self.sample_rate} Гц)")
        except Exception as e:
            print(f"Ошибка при сохранении: {e}")
            
if __name__ == "__main__":
    lines = [
        "-------",
        "Съешь ещё этих мягких французских булок, да выпей же чаю» (русский),",
        "The quick brown fox jumps over the lazy dog(english). 1234567890",
        "-------"
    ]

    # Объединяем в одно сообщение через перевод строки (без дополнительных пауз)
    text = "\n".join(lines)

    # Создаём payload и сигнал
    payload = Payload(text)
    rtty = RTTYSignal(
        payload,
        baud=45.45,
        mark_freq=1170,
        space_freq=1000,
        amplitude=10000
    )

    rtty.modulate()
    rtty.play()  # воспроизводим единым потоком (без пауз между строками)
    rtty.save_wav("rtty_message") # Сохраняем в WAV‑файл
