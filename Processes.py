from music21 import converter, instrument, note, chord, stream

class MIDIProcessor:
    """Processes MIDI files into token sequences using music21（改进音乐结构解析能力）"""
    def __init__(self, time_resolution=0.01):
        # 保持与原始相同的词汇定义
        self.note_on_offset = 0
        self.note_off_offset = 128
        self.time_shift_offset = 256
        self.vocab_size = 384
        
        # 新增参数：时间分辨率（秒/单位时间）
        self.time_resolution = time_resolution  # 默认0.01秒（10ms）

    def midi_to_sequence(self, midi_path):
        """改进版MIDI解析，支持多音轨和弦检测"""
        score = converter.parse(midi_path)
        sequence = []
        last_time = 0.0  # 使用绝对时间（秒）

        # 合并所有音符事件并按时间排序
        events = []
        for part in score.parts:
            # 提取乐器信息（可选）
            instr = part.getElementsByClass(instrument.Instrument).first()
            
            for element in part.flat.notesAndRests:
                # 处理休止符
                if isinstance(element, note.Rest):
                    events.append(('rest', element.offset, element.duration.quarterLength))
                
                # 处理单个音符
                elif isinstance(element, note.Note):
                    events.append(('note_on', element.offset, element.pitch.midi))
                    events.append(('note_off', element.offset + element.duration.quarterLength, element.pitch.midi))
                
                # 处理和弦
                elif isinstance(element, chord.Chord):
                    for p in element.pitches:
                        events.append(('note_on', element.offset, p.midi))
                        events.append(('note_off', element.offset + element.duration.quarterLength, p.midi))

        # 按时间排序所有事件
        events.sort(key=lambda x: x[1])

        # 生成token序列
        for event_type, event_time, value in events:
            # 计算时间差
            delta = event_time - last_time
            
            # 生成时间间隔token（改进：支持高精度时间）
            if delta > 0:
                time_units = int(round(delta / self.time_resolution))
                while time_units > 0:
                    units = min(time_units, 127)  # 单次最多127单位
                    sequence.append(self.time_shift_offset + units)
                    time_units -= units
            
            # 处理事件
            if event_type == 'note_on':
                sequence.append(self.note_on_offset + value)
            elif event_type == 'note_off':
                sequence.append(self.note_off_offset + value)
            
            last_time = event_time
        
        return sequence

    def sequence_to_midi(self, sequence, output_path, tempo=60):
        """改进版MIDI生成，支持动态速度"""
        s = stream.Stream()
        current_time = 0.0  # 当前时间（秒）
        active_notes = {}  # 跟踪活跃音符：{pitch: start_time}

        for token in sequence:
            if self.note_on_offset <= token < self.note_off_offset:
                pitch = token - self.note_on_offset
                # 记录音符开始时间
                active_notes[pitch] = current_time
                
            elif self.note_off_offset <= token < self.time_shift_offset:
                pitch = token - self.note_off_offset
                if pitch in active_notes:
                    # 创建音符对象
                    n = note.Note(pitch)
                    n.offset = active_notes[pitch]  # 设置绝对开始时间
                    n.duration.quarterLength = (current_time - active_notes[pitch]) * tempo / 60
                    s.insert(n)
                    del active_notes[pitch]
                    
            elif self.time_shift_offset <= token < self.vocab_size:
                units = token - self.time_shift_offset
                current_time += units * self.time_resolution

        # 添加默认钢琴音色
        s.insert(0, instrument.Piano())
        
        # 导出MIDI（改进：保留时间精度）
        s.write('midi', fp=output_path)
        return s

