import configparser
import re
import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
from collections import defaultdict

# 1. 首先创建主窗口（但先隐藏）
root = tk.Tk()
root.withdraw()

# 2. 创建Tkinter变量
rules_file_path = tk.StringVar()
mode_var = tk.StringVar(value="vccv-cvvc")
generation_mode_var = tk.StringVar(value="none")
language_var = tk.StringVar()

# 3. 其他全局变量
current_language = "zh"
text = {}
config_file = "reclistgen++_config.ini"
language_file = "languages.ini"
recording_list = []
oto_entries = []
syllable_counts = {}
generation_mode_map = {}
# 生成模式映射表，将在加载语言文件后初始化

# 工具函数
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(_nsre, s)]

def get_text(key):
    return text.get(key, key)

# 核心功能类
class RuleParser:
    @staticmethod
    def parse(rule_file):
        rules = {}
        config = configparser.ConfigParser(allow_no_value=True)
        try:
            config.read(rule_file, encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Failed to load config file: {e}")
        if 'RULE' not in config:
            raise ValueError("Missing [RULE] section in rule file")
        
        pattern = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')
        for syllable, value in config['RULE'].items():
            components = [c.strip().strip('"') for c in pattern.split(value)]
            if len(components) != 4:
                raise ValueError(f"Invalid rule format for {syllable}")
            consonant, onset, transition, coda = components
            transition_list = [[t.strip()] for t in components[2].split(',')]
            rules[syllable] = {
                'consonant': consonant,
                'onset': onset,
                'transition': transition_list,
                'coda': coda
            }
        return rules

    @staticmethod
    def generate_standard_entries(rules, syllables, max_syllables=8):
        # 解析五大部分
        start_cv = set()
        cv = set()
        transitions = set()
        end_vr = set()
        vc = set()
        for syl, parts in rules.items():
            onset = parts.get('onset')
            consonant = parts.get('consonant')
            transition = parts.get('transition')
            coda = parts.get('coda')
            if onset:
                start_cv.add(onset)
            if consonant:
                cv.add(consonant)
            if transition:
                transitions.add(transition.replace('"','').strip())
            if coda:
                end_vr.add(coda)
        # VC部覆盖（简化：所有coda-consonant组合）
        for syl1, parts1 in rules.items():
            coda1 = parts1.get('coda')
            for syl2, parts2 in rules.items():
                consonant2 = parts2.get('consonant')
                if coda1 and consonant2:
                    vc.add((coda1, consonant2))
        # 合并生成句子（此处为简化示例，实际可优化）
        # 收集所有需要覆盖的音素和VC组合
        all_phonemes_to_cover = set()
        all_phonemes_to_cover.update(start_cv)
        all_phonemes_to_cover.update(cv)
        all_phonemes_to_cover.update(transitions)
        all_phonemes_to_cover.update(end_vr)
        all_phonemes_to_cover.update(vc) # VC组合是元组，直接添加

        # 构建一个映射，从音素到包含该音素的拼音列表
        phoneme_to_pinyins = defaultdict(list)
        for pinyin, parts in rules.items():
            onset = parts.get('onset')
            consonant = parts.get('consonant')
            transition = parts.get('transition')
            coda = parts.get('coda')
            if onset: phoneme_to_pinyins[onset].append(pinyin)
            if consonant: phoneme_to_pinyins[consonant].append(pinyin)
            if transition:
                phoneme_to_pinyins[transition.replace('"','').strip()].append(pinyin)
            if coda: phoneme_to_pinyins[coda].append(pinyin)
            # VC组合
            for next_pinyin, next_parts in rules.items():
                next_onset = next_parts.get('onset')
                next_consonant = next_parts.get('consonant')
                next_transition = next_parts.get('transition')
                next_coda = next_parts.get('coda')
                if coda and next_consonant:
                    phoneme_to_pinyins[(coda, next_consonant)].append(f"{pinyin}_{next_pinyin}")

        # 贪婪算法生成覆盖所有音素的拼音句子
        covered_phonemes = set()
        result_entries = []
        current_sentence_pinyins = []

        # 优先覆盖未被覆盖的音素
        valid_phonemes_to_cover = [ph for ph in all_phonemes_to_cover if phoneme_to_pinyins[ph]]
        sorted_phonemes_to_cover = sorted(list(valid_phonemes_to_cover), key=lambda x: len(phoneme_to_pinyins[x]))

        for target_phoneme in sorted_phonemes_to_cover:
            if target_phoneme not in covered_phonemes:
                candidate_pinyins = phoneme_to_pinyins[target_phoneme]
                if not candidate_pinyins:
                    continue

                selected_pinyin = candidate_pinyins[0]

                if isinstance(target_phoneme, tuple):
                    pinyins_in_sentence = selected_pinyin.split('_')
                else:
                    pinyins_in_sentence = [selected_pinyin]

                if not current_sentence_pinyins:
                    current_sentence_pinyins.extend(pinyins_in_sentence)
                elif len(current_sentence_pinyins) + len(pinyins_in_sentence) <= max_syllables:
                    current_sentence_pinyins.extend(pinyins_in_sentence)
                else:
                    result_entries.append('_'.join(current_sentence_pinyins))
                    current_sentence_pinyins = pinyins_in_sentence

                for pinyin_in_sentence in pinyins_in_sentence:
                    if pinyin_in_sentence in rules:
                        onset = rules[pinyin_in_sentence].get('onset')
                        consonant = rules[pinyin_in_sentence].get('consonant')
                        transition = rules[pinyin_in_sentence].get('transition')
                        coda = rules[pinyin_in_sentence].get('coda')
                        if onset: covered_phonemes.add(onset)
                        if consonant: covered_phonemes.add(consonant)
                        if transition: covered_phonemes.add(transition.strip('"'))
                        if coda: covered_phonemes.add(coda)
                    for i in range(len(pinyins_in_sentence) - 1):
                        current_pinyin = pinyins_in_sentence[i]
                        next_pinyin = pinyins_in_sentence[i+1]
                        if current_pinyin in rules and next_pinyin in rules:
                            coda = rules[current_pinyin].get('coda')
                            consonant = rules[next_pinyin].get('consonant')
                            if coda and consonant:
                                covered_phonemes.add((coda, consonant))

        if current_sentence_pinyins:
            result_entries.append('_'.join(current_sentence_pinyins))

        return result_entries

    @staticmethod 
    def _build_coda_consonant_graph(rules): 
        """构建音素依赖关系图 - 优化版本"""
        valid_consonants = set()
        valid_codas = set()
        coda_to_syllables = defaultdict(list)
        consonant_to_syllables = defaultdict(list)

        for syl, parts in rules.items():
            if parts['consonant']:
                valid_consonants.add(parts['consonant'])
                consonant_to_syllables[parts['consonant']].append(syl)
            if parts['coda']:
                valid_codas.add(parts['coda'])
                coda_to_syllables[parts['coda']].append(syl)

        graph = defaultdict(dict)
        for coda in valid_codas:
            for consonant in valid_consonants:
                graph[coda][consonant] = len(coda_to_syllables[coda]) * len(consonant_to_syllables[consonant])
        return graph

    @staticmethod
    def _cover_critical_combinations(standard_entries, rules, syllables, coverage):
        """优先覆盖关键音素组合"""
        for syl in syllables:
            if syl not in coverage['start']:
                standard_entries.add(syl)
            if syl not in coverage['middle']:
                standard_entries.add(syl)

        all_codas = {parts['coda'] for parts in rules.values() if parts['coda']}
        missing_end_codas = all_codas - coverage['end_coda']
        syllables_with_coda = defaultdict(list)
        
        for syl, parts in rules.items():
            if parts['coda']:
                syllables_with_coda[parts['coda']].append(syl)

        for coda in missing_end_codas:
            if syllables_with_coda[coda]:
                standard_entries.add(syllables_with_coda[coda][0])

    @staticmethod
    def _get_remaining_combinations(coverage, coda_consonant_graph):
        """获取未覆盖的音素组合 - 优化版本"""
        remaining = defaultdict(set)
        covered_pairs = coverage['coda_consonant']

        for coda, consonants_dict in coda_consonant_graph.items():
            if not coda:
                continue
                
            all_consonants = set(consonants_dict.keys())
            covered_consonants = set(cons for c, cons in covered_pairs if c == coda)
            uncovered_consonants = all_consonants - covered_consonants

            if uncovered_consonants:
                remaining[coda] = uncovered_consonants
                print(f"尾音 '{coda}' 还有 {len(uncovered_consonants)} 个未覆盖的辅音组合需要生成")

        total_uncovered = sum(len(consonants) for consonants in remaining.values())
        if total_uncovered > 0:
            print(f"总计有 {total_uncovered} 个尾音-辅音组合需要由标准生成算法覆盖")
        else:
            print("所有尾音-辅音组合已被强制生成模式完全覆盖")
        
        return remaining

    @staticmethod
    def _greedy_coverage(standard_entries, remaining_combos, rules):
        """贪心算法实现组合覆盖"""
        while remaining_combos:
            best_coda = None
            best_consonant = None
            max_score = -1

            for coda, consonants in remaining_combos.items():
                for consonant in consonants:
                    score = RuleParser._calculate_coverage_score(coda, consonant, 
                        defaultdict(list, {c: [s for s in rules if rules[s]['coda'] == c]}),
                        defaultdict(list, {c: [s for s in rules if rules[s]['consonant'] == c]}))
                    if score > max_score:
                        max_score = score
                        best_coda = coda
                        best_consonant = consonant

            if best_coda and best_consonant:
                entry = f"{random.choice([s for s in rules if rules[s]['coda'] == best_coda])}_{random.choice([s for s in rules if rules[s]['consonant'] == best_consonant])}"
                standard_entries.add(entry)
                remaining_combos[best_coda].discard(best_consonant)
                if not remaining_combos[best_coda]:
                    del remaining_combos[best_coda]

    @staticmethod
    def _add_missing_syllables(standard_entries, syllables, coverage, rules, settings):
        """补充未覆盖的独立音节"""
        missing_start = [s for s in syllables if s not in coverage['start']]
        missing_middle = [s for s in syllables if s not in coverage['middle']]
        
        if missing_start:
            standard_entries.update(random.sample(missing_start, min(3, len(missing_start))))
        if missing_middle:
            standard_entries.update(random.sample(missing_middle, min(3, len(missing_middle))))
    
    @staticmethod 
    def _build_coda_consonant_graph(rules): 
        """构建音素依赖关系图 - 优化版本"""
        # 预先提取所有有效的辅音和尾音，避免重复计算
        valid_consonants = set()
        valid_codas = set()
        coda_to_syllables = defaultdict(list)
        consonant_to_syllables = defaultdict(list)
        
        # 第一次遍历：收集所有有效的辅音和尾音
        for syl, parts in rules.items():
            if parts['consonant']:
                valid_consonants.add(parts['consonant'])
                consonant_to_syllables[parts['consonant']].append(syl)
            if parts['coda']:
                valid_codas.add(parts['coda'])
                coda_to_syllables[parts['coda']].append(syl)
        
        # 构建图 - 直接使用已知的有效辅音和尾音
        graph = defaultdict(dict)
        for coda in valid_codas:
            for consonant in valid_consonants:
                # 只在图中存储有效的组合，并记录组合的频率
                graph[coda][consonant] = len(coda_to_syllables[coda]) * len(consonant_to_syllables[consonant])
        
        return graph
    
    @staticmethod
    def _cover_critical_combinations(standard_entries, rules, syllables, coverage):
        """优先覆盖关键音素组合"""
        # 1. 检查并添加未覆盖的单个音节 (起始和中间)
        for syl in syllables:
            if syl not in coverage['start']:
                standard_entries.add(syl) # 优先添加单个音节以覆盖起始
            if syl not in coverage['middle']:
                standard_entries.add(syl)
        
        # 2. 检查未覆盖的结尾 Coda
        all_codas = {parts['coda'] for parts in rules.values() if parts['coda']}
        missing_end_codas = all_codas - coverage['end_coda']
        
        syllables_with_coda = defaultdict(list)
        for syl, parts in rules.items():
            if parts['coda']:
                syllables_with_coda[parts['coda']].append(syl)
                
        for coda in missing_end_codas:
            # 需要生成以带这个 coda 的音节结尾的条目
            if syllables_with_coda[coda]:
                syl_with_coda = syllables_with_coda[coda][0]
                standard_entries.add(syl_with_coda) # 添加单个音节作为结尾
    
    @staticmethod
    def _get_remaining_combinations(coverage, coda_consonant_graph):
        """获取未覆盖的音素组合 - 优化版本
        考虑强制生成模式已覆盖的音素组合，确保标准生成只填补未覆盖的部分
        """
        remaining = defaultdict(set)
        covered_pairs = coverage['coda_consonant']
        
        # 使用集合操作更高效地找出未覆盖的组合
        for coda, consonants_dict in coda_consonant_graph.items():
            # 只处理有效的尾音
            if not coda:
                continue
                
            # 获取当前尾音可以连接的所有辅音
            all_consonants = set(consonants_dict.keys())
            
            # 找出已覆盖的辅音 - 这里考虑了强制生成模式已覆盖的组合
            covered_consonants = set(cons for c, cons in covered_pairs if c == coda)
            
            # 计算未覆盖的辅音（差集操作）
            uncovered_consonants = all_consonants - covered_consonants
            
            # 只添加有效的未覆盖组合
            if uncovered_consonants:
                remaining[coda] = uncovered_consonants
                print(f"尾音 '{coda}' 还有 {len(uncovered_consonants)} 个未覆盖的辅音组合需要生成")
        
        # 输出总体覆盖情况
        total_uncovered = sum(len(consonants) for consonants in remaining.values())
        if total_uncovered > 0:
            print(f"总计有 {total_uncovered} 个尾音-辅音组合需要由标准生成算法覆盖")
        else:
            print("所有尾音-辅音组合已被强制生成模式完全覆盖")
            
        return remaining
    
    @staticmethod
    def _calculate_coverage_score(coda, consonant, coda_syllables, consonant_syllables):
        """计算组合的覆盖价值 - 优化版本"""
        # 直接使用预计算的音节列表长度，避免遍历整个规则集
        return len(coda_syllables.get(coda, [])) * len(consonant_syllables.get(consonant, []))
    
    @staticmethod
    def _find_best_syllable(coda_syllables, consonant_syllables, coda=None, consonant=None):
        """找到最适合的音节 - 优化版本"""
        if coda and coda in coda_syllables and coda_syllables[coda]:
            return coda_syllables[coda][0]
        elif consonant and consonant in consonant_syllables and consonant_syllables[consonant]:
            return consonant_syllables[consonant][0]
        return None
    
    @staticmethod
    def _update_remaining_combos(remaining_combos, combo_to_remove):
        """更新剩余组合列表 - 优化版本"""
        # 由于我们已经重构了贪心算法，这个方法实际上不再需要
        # 保留此方法是为了兼容性，但它现在只是一个空操作
        return remaining_combos
    
    @staticmethod
    def _greedy_coverage(entries, remaining_combos, rules):
        """贪心算法实现最优组合选择 - 优化版本""" 
        # 预计算所有组合的分数，避免重复计算
        combo_scores = {}
        coda_syllables = defaultdict(list)
        consonant_syllables = defaultdict(list)
        
        # 预处理：建立音素到音节的映射
        for syl, parts in rules.items():
            if parts['coda']:
                coda_syllables[parts['coda']].append(syl)
            if parts['consonant']:
                consonant_syllables[parts['consonant']].append(syl)
        
        # 预计算所有组合的分数
        for coda, consonants in remaining_combos.items():
            for cons in consonants:
                # 使用音节数量作为分数，避免遍历整个规则集
                score = len(coda_syllables[coda]) * len(consonant_syllables[cons])
                combo_scores[(coda, cons)] = score
        
        # 按分数降序排序所有组合
        sorted_combos = sorted(
            [(coda, cons) for coda, consonants in remaining_combos.items() for cons in consonants],
            key=lambda x: combo_scores.get(x, 0),
            reverse=True
        )
        
        # 贪心选择最高分数的组合
        covered_combos = set()
        for coda, cons in sorted_combos:
            # 如果组合已被覆盖，跳过
            if (coda, cons) in covered_combos:
                continue
                
            # 找到最适合的音节
            syl1 = coda_syllables[coda][0] if coda_syllables[coda] else None
            syl2 = consonant_syllables[cons][0] if consonant_syllables[cons] else None
            
            if syl1 and syl2:
                entries.add(f"{syl1}_{syl2}")
                covered_combos.add((coda, cons))
                
                # 更新已覆盖的组合集合
                # 这里可以添加额外的逻辑来标记那些通过这个条目间接覆盖的组合
        
        # 不再需要循环更新remaining_combos，因为我们已经处理了所有组合
    
    @staticmethod
    def _add_missing_syllables(standard_entries, syllables, coverage, rules, settings):
        """确保所有音节至少出现一次，并确保所有音素类型都被覆盖 - 增强版本"""
        # 收集已经在条目中出现的所有音节
        covered_syllables = set()
        for entry in standard_entries:
            covered_syllables.update(entry.split('_'))
        
        # 使用集合差集操作找出未覆盖的音节
        missing_syllables = set(syllables) - covered_syllables
        
        # 将未覆盖的音节添加到结果中
        standard_entries.update(missing_syllables)
        print(f"添加了 {len(missing_syllables)} 个未覆盖的音节")
        
        # --- 新增：确保所有音素类型都被覆盖 ---
        # 1. 检查所有开头整音（前面是空的整音）
        for syl, rule in rules.items():
            if rule['onset'] and syl not in coverage['start']:
                standard_entries.add(syl)
                coverage['start'].add(syl)
                print(f"添加开头整音: {syl}")
        
        # 2. 检查所有transition元素（介音，双元音等）
        for syl, rule in rules.items():
            for idx, trans_group in enumerate(rule['transition']):
                if trans_group and trans_group[0] and idx not in coverage['transition'][syl]:
                    standard_entries.add(syl)
                    if syl not in coverage['transition']:
                        coverage['transition'][syl] = set()
                    coverage['transition'][syl].add(idx)
                    print(f"添加transition元素: {syl} (索引 {idx})")
        
        # 3. 检查所有coda到R的连接
        for syl, rule in rules.items():
            if rule['coda']:
                # 检查是否已经有这个coda到R的连接
                coda_r_exists = False
                for e in standard_entries:
                    if '_' in e and e.endswith('_R'):
                        prev_syl = e.split('_')[-2]
                        if prev_syl in rules and rules[prev_syl]['coda'] == rule['coda']:
                            coda_r_exists = True
                            break
                
                if not coda_r_exists:
                    standard_entries.add(f"{syl}_R")
                    print(f"添加coda到R连接: {syl}_R")
        
        # 4. 确保前面有R的拼音正确标记为开头整音
        for entry in list(standard_entries):
            parts = entry.split('_')
            for i in range(1, len(parts)):
                if parts[i-1] == 'R' and parts[i] in rules:
                    # 确保这个音节也作为开头整音出现
                    if parts[i] not in coverage['start']:
                        standard_entries.add(parts[i])
                        coverage['start'].add(parts[i])
                        print(f"添加R后的开头整音: {parts[i]}")
        
        # 5. 确保所有音素连接组合都被覆盖（根据当前模式）
        mode = settings.get('mode', 'vccv-cvvc')
        all_codas = {parts['coda'] for parts in rules.values() if parts['coda']}
        all_consonants = {parts['consonant'] for parts in rules.values() if parts['consonant']}
        all_onsets = {parts['onset'] for parts in rules.values() if parts['onset']}
        
        if mode == "cvc":
            # CVC模式：确保所有整音(onset)到辅音(consonant)的连接都被覆盖
            for onset in all_onsets:
                for consonant in all_consonants:
                    # 检查这个onset-consonant组合是否已经被覆盖
                    # 注意：CVC模式下我们仍然使用coda_consonant集合来跟踪，但实际存储的是onset-consonant对
                    if (onset, consonant) not in coverage.get('onset_consonant', set()):
                        # 找到具有这个onset的音节
                        onset_syllable = None
                        for syl, parts in rules.items():
                            if parts['onset'] == onset:
                                onset_syllable = syl
                                break
                        
                        # 找到具有这个consonant的音节
                        consonant_syllable = None
                        for syl, parts in rules.items():
                            if parts['consonant'] == consonant:
                                consonant_syllable = syl
                                break
                        
                        if onset_syllable and consonant_syllable:
                            entry = f"{onset_syllable}_{consonant_syllable}"
                            standard_entries.add(entry)
                            if 'onset_consonant' not in coverage:
                                coverage['onset_consonant'] = set()
                            coverage['onset_consonant'].add((onset, consonant))
                            print(f"添加onset-consonant连接: {entry} ({onset} {consonant})")
        else:
            # VCCV-CVVC和VCV模式：确保所有coda到consonant的连接都被覆盖
            for coda in all_codas:
                for consonant in all_consonants:
                    if (coda, consonant) not in coverage['coda_consonant']:
                        # 找到具有这个coda的音节
                        coda_syllable = None
                        for syl, parts in rules.items():
                            if parts['coda'] == coda:
                                coda_syllable = syl
                                break
                        
                        # 找到具有这个consonant的音节
                        consonant_syllable = None
                        for syl, parts in rules.items():
                            if parts['consonant'] == consonant:
                                consonant_syllable = syl
                                break
                        
                        if coda_syllable and consonant_syllable:
                            entry = f"{coda_syllable}_{consonant_syllable}"
                            standard_entries.add(entry)
                            coverage['coda_consonant'].add((coda, consonant))
                            print(f"添加coda-consonant连接: {entry} ({coda} {consonant})")
        
        return standard_entries

    @staticmethod
    def generate_recording_list(rules, settings):
        """
        生成录音列表，结合强制生成模式和标准覆盖生成。
        强制生成的条目保持原样，标准生成的条目会进行合并以优化长度。
        """
        max_syllables = settings['max_syllables_per_sentence']
        generation_mode = settings.get('generation_mode', 'none')
        auto_merge = settings.get('auto_merge', True)  # 新增：是否自动合并条目
        syllables = list(rules.keys())

        # 初始化覆盖率跟踪字典
        coverage = {
            'start': set(),
            'middle': set(),
            'transition': defaultdict(set),
            'end_coda': set(),
            'coda_consonant': set()
        }

        # --- 第一步：强制生成模式 ---
        forced_entries = [] # 存储强制生成的原始条目
        mode_handlers = {
            'repeat': lambda: RuleParser.handle_repeat(rules, syllables, forced_entries, coverage),
            'interval': lambda: RuleParser.handle_interval(rules, syllables, forced_entries, coverage, max_syllables),
            'sequence': lambda: RuleParser.handle_sequence(rules, syllables, forced_entries, coverage, max_syllables)
        }

        # 如果指定了强制生成模式，则执行
        if generation_mode in mode_handlers:
            print(f"执行强制生成模式: {generation_mode}") # 添加日志
            mode_handlers[generation_mode]()
        else:
            print("未指定或未识别强制生成模式，跳过强制生成。") # 添加日志

        # 强制生成的条目列表
        # 复制一份以确保原始列表不被修改
        final_forced_entries = forced_entries.copy()
        print(f"强制生成条目数: {len(final_forced_entries)}") # 添加日志

        # --- 新增：应用合并策略 ---
        merge_strategy = {
            'repeat': RuleParser._merge_repeat_entries,
            'interval': RuleParser._merge_interval_entries,
            'sequence': RuleParser._merge_sequence_entries
        }
        
        # 应用合并策略（关键改进点）
        # 任何强制生成模式都不应该被合并，以保留每种模式设计的特定采样模式
        if generation_mode in merge_strategy and auto_merge and generation_mode == 'none':
            print(f"应用{generation_mode}模式智能合并策略...") # 添加日志
            merged_forced = merge_strategy[generation_mode](final_forced_entries, max_syllables, rules)
            final_forced_entries = merged_forced
            print(f"合并后强制生成条目数: {len(final_forced_entries)}") # 添加日志
        elif generation_mode != 'none':
            print(f"{generation_mode}模式不进行合并，保留该模式的特定采样模式") # 添加日志

        # --- 第二步：标准条目生成 (基于覆盖率) ---
        # 生成标准条目以覆盖强制生成未覆盖的音素组合
        print("开始生成标准条目...") # 添加日志
        # 收集强制生成模式已覆盖的音素
        initial_covered_phonemes = set()
        for entry in final_forced_entries:
            pinyins = entry.split('_')
            for i, pinyin in enumerate(pinyins):
                if pinyin in rules:
                    onset, consonant, transition, coda = rules[pinyin]
                    if onset: initial_covered_phonemes.add(onset)
                    if consonant: initial_covered_phonemes.add(consonant)
                    if transition: initial_covered_phonemes.add(transition.strip('"'))
                    if coda: initial_covered_phonemes.add(coda)
                # 检查VC组合
                if i < len(pinyins) - 1:
                    next_pinyin = pinyins[i+1]
                    if pinyin in rules and next_pinyin in rules:
                        coda = rules[pinyin][3]
                        consonant = rules[next_pinyin][1]
                        if coda and consonant:
                            initial_covered_phonemes.add((coda, consonant))

        standard_entries = RuleParser.generate_standard_entries(rules, syllables, max_syllables,
            settings,
            initial_covered_phonemes=initial_covered_phonemes)
        print(f"生成标准条目数 (合并前): {len(standard_entries)}") # 添加日志

        # --- 第三步：合并标准条目 ---
        # 只对标准生成的条目进行合并，以减少总条目数并满足最大音节限制
        if auto_merge:
            print("开始合并标准条目...") # 添加日志
            merged_standard_entries = RuleParser._merge_standard_entries(standard_entries, max_syllables, rules)
            print(f"合并后标准条目数: {len(merged_standard_entries)}") # 添加日志
            unmerged_standard_entries = merged_standard_entries
        else:
            print("标准条目合并步骤已禁用。") # 添加日志
            unmerged_standard_entries = standard_entries # 直接使用未合并的列表

        # --- 第四步：组合最终列表 ---
        # 最终列表 = 强制生成的条目 (已合并) + 合并后的标准条目
        # 注意：强制生成的条目应该保持原样，不应该与标准条目混合或去重
        # 这样可以确保强制生成的录音命名不会混入标准采样中
        final_list = []
        
        # 首先添加强制生成的条目（保持原顺序）
        if final_forced_entries:
            print(f"将强制生成条目放在列表开头...") # 添加日志
            final_list.extend(final_forced_entries)
        
        # 然后添加那些不在强制生成条目中的标准条目（按自然排序）
        forced_entries_set = set(final_forced_entries)
        standard_entries_to_add = []
        for entry in unmerged_standard_entries:
            if entry not in forced_entries_set:
                standard_entries_to_add.append(entry)
        
        # 对标准条目进行排序后添加到最终列表
        if standard_entries_to_add:
            sorted_standard_entries = sorted(standard_entries_to_add, key=natural_sort_key)
            final_list.extend(sorted_standard_entries)
                
        print(f"最终录音列表总条目数: {len(final_list)}") # 添加日志

        # --- 第五步：计算音节统计 ---
        syllable_counts = defaultdict(int)
        for recording in final_list:
            for syllable in recording.split('_'):
                if syllable != 'R' and syllable in rules: # 确保是有效音节
                    syllable_counts[syllable] += 1

        # 返回最终列表、所有音节列表和音节计数
        return final_list, syllables, syllable_counts

# -------------------------- 新增辅助函数 --------------------------

    @staticmethod
    def update_coverage(entry, rules, coverage, position='middle'):
        """分析音节并更新覆盖状态"""
        tokens = entry.split('_')
        total = len(tokens)
        
        for i, token in enumerate(tokens):
            if token == 'R':
                continue
                
            rule = rules[token]
            # 1. 处理开头位置
            if i == 0:
                coverage['start'].add(token)
            
            # 2. 处理中间位置
            if i > 0:
                coverage['middle'].add(token)
                
            # 3. 处理结尾coda
            if i == total - 1 and rule['coda']:
                coverage['end_coda'].add(rule['coda'])
                
            # 4. 处理transition
            for idx, trans in enumerate(rule['transition']):
                if trans and trans[0]:  # 确保transition有效
                    coverage['transition'][token].add(idx)
                    
            # 5. 处理coda到辅音的组合
            if i < total - 1:
                next_rule = rules.get(tokens[i+1], {})
                next_consonant = next_rule.get('consonant', '')
                if rule['coda'] and next_consonant:
                    coverage['coda_consonant'].add((rule['coda'], next_consonant))
    
    @staticmethod
    def handle_repeat(rules, syllables, entries, coverage):
        """Repeat模式生成器"""
        for syl in syllables:
            entry = f"{syl}_{syl}_{syl}"
            entries.append(entry)
            RuleParser.update_coverage(entry, rules, coverage)
            
            # 特殊标记开头和结尾
            coverage['start'].add(syl)
            coverage['middle'].add(syl)
            if rules[syl]['coda']:
                coverage['end_coda'].add(rules[syl]['coda'])
    
    @staticmethod
    def handle_interval(rules, syllables, entries, coverage, max_syl):
        """Interval模式生成器 - 严格遵守一拼音一休止符'R'的规则，但最后一个R会被省略，且确保首音节始终是拼音"""
        # 创建严格的一拼音一休止符模式
        pattern = []
        for syl in syllables:
            pattern.append(syl)  # 添加音节
            pattern.append("R")  # 添加休止符
        
        current = []
        i = 0
        while i < len(pattern):
            # 获取当前音节
            syllable = pattern[i] if i < len(pattern) else None
            
            # 如果添加这个音节会超出最大音节数，先保存当前条目
            if syllable and (len(current) + 1 > max_syl):
                if current:
                    # 如果最后一个是R，则省略它
                    if current[-1] == "R":
                        current = current[:-1]
                    entry = '_'.join(current)
                    entries.append(entry)
                    RuleParser.update_coverage(entry, rules, coverage)
                    current = []
            
            # 添加音节到当前条目
            if syllable:
                # 确保当前条目为空时，首个添加的音节不是R
                if not current and syllable == "R":
                    # 跳过这个R，直接移动到下一个元素
                    i += 1
                    continue
                    
                current.append(syllable)
                i += 1  # 移动到下一个元素（休止符）
                
                # 只有当不是最后一个音节时，才添加休止符
                if i < len(pattern) and pattern[i] == "R" and len(current) < max_syl:
                    current.append("R")
                    i += 1  # 移动到下一个元素（下一个音节）
                    
                    # 避免连续的R - 如果下一个也是R，跳过它
                    while i < len(pattern) and pattern[i] == "R":
                        i += 1
            else:
                i += 1  # 如果当前元素为None，移动到下一个
        
        # 保存最后一个条目
        if current:
            # 无论奇偶性如何，如果最后一个是R，都省略它
            if current[-1] == "R":
                current = current[:-1]
            # 再次确保没有以R开头的条目
            if current and current[0] == "R":
                current = current[1:]
            # 只有当条目非空时才添加
            if current:
                entry = '_'.join(current)
                entries.append(entry)
                RuleParser.update_coverage(entry, rules, coverage)
    
    @staticmethod
    def handle_sequence(rules, syllables, entries, coverage, max_syl):
        """Sequence模式生成器 - 简单按顺序组合拼音，不进行分组"""
        # 直接处理所有拼音，不需要分组
        current = []
        
        for syl in syllables:
            if len(current) < max_syl:
                current.append(syl)
            else:
                # 当达到最大音节数时，创建一个条目并重新开始
                entry = '_'.join(current)
                entries.append(entry)
                RuleParser.update_coverage(entry, rules, coverage)
                current = [syl]  # 开始新的条目
        
        # 处理最后一组
        if current:
            entry = '_'.join(current)
            entries.append(entry)
            RuleParser.update_coverage(entry, rules, coverage)
            
        # 记录覆盖情况
        for entry in entries:
            parts = entry.split('_')
            if parts:
                coverage['start'].add(parts[0])
                
                for i in range(1, len(parts)):
                    coverage['middle'].add(parts[i])
    
    @staticmethod
    def _merge_repeat_entries(entries, max_syllables, rules):
        """合并重复模式生成的条目
        重复模式的特点是每个条目都是同一个音节重复三次，例如 "syl_syl_syl"。
        合并时需要保持这种重复模式的特性，同时优化总条目数。
        """
        # 按音节分组，确保相同音节的重复条目被放在一起处理
        syllable_groups = defaultdict(list)
        for entry in entries:
            # 提取重复条目的基础音节（假设格式为 syl_syl_syl）
            parts = entry.split('_')
            if len(parts) > 0 and all(part == parts[0] for part in parts):
                syllable_groups[parts[0]].append(entry)
            else:
                # 对于不符合重复模式的条目，使用整个条目作为键
                syllable_groups[entry].append(entry)
        
        # 合并后的条目列表
        merged = []
        current = []
        
        # 按自然排序处理分组后的条目
        for syllable in sorted(syllable_groups.keys(), key=natural_sort_key):
            entries_in_group = syllable_groups[syllable]
            
            for entry in entries_in_group:
                parts = entry.split('_')
                # 检查是否可以添加到当前条目中
                if len(current) + len(parts) <= max_syllables:
                    current.extend(parts)
                else:
                    # 当前条目已满，添加到结果并开始新条目
                    if current:
                        merged.append('_'.join(current))
                    current = parts
        
        # 添加最后一个条目
        if current:
            merged.append('_'.join(current))
            
        return merged
        
    @staticmethod
    def _merge_interval_entries(entries, max_syllables, rules):
        """合并间隔模式生成的条目
        间隔模式的特点是音节之间有R作为间隔，例如 "syl_R_syl_R"。
        合并时需要严格保持一拼音一休止符的模式，同时尽量达到最大字数限制。
        """
        # 不进行合并，保持原始的间隔模式条目
        # 间隔模式的条目应该在生成时就已经按照一拼音一休止符的规则生成
        # 合并可能会破坏这种模式，因此直接返回原始条目
        print("间隔模式条目不进行合并，保持原始的一拼音一休止符模式")
        return entries
        
    @staticmethod
    def _merge_sequence_entries(entries, max_syllables, rules):
        """合并序列模式生成的条目
        序列模式的特点是按辅音分组的音节序列，例如 "syl1_syl2_syl3"，其中所有音节共享相同的辅音。
        合并时需要尽量保持这种分组特性，同时优化总条目数。
        """
        # 按辅音分组
        consonant_groups = defaultdict(list)
        
        for entry in entries:
            parts = entry.split('_')
            if len(parts) > 0:
                # 尝试获取第一个音节的辅音作为分组键
                first_syl = parts[0]
                consonant = rules.get(first_syl, {}).get('consonant', '')
                consonant_groups[consonant or 'other'].append(entry)
        
        # 合并后的条目列表
        merged = []
        current = []
        
        # 按辅音分组处理，保持相同辅音的音节尽量在一起
        for consonant in sorted(consonant_groups.keys(), key=natural_sort_key):
            entries_in_group = consonant_groups[consonant]
            
            for entry in sorted(entries_in_group, key=natural_sort_key):
                parts = entry.split('_')
                # 检查是否可以添加到当前条目中
                if len(current) + len(parts) <= max_syllables:
                    current.extend(parts)
                else:
                    # 当前条目已满，添加到结果并开始新条目
                    if current:
                        merged.append('_'.join(current))
                    current = parts
        
        # 添加最后一个条目
        if current:
            merged.append('_'.join(current))
            
        return merged
        
    @staticmethod
    def _is_phoneme_covered_elsewhere(syllable, rules, entries, current_entry=None):
        """检查一个音节的音素是否在其他条目中已被覆盖
        
        参数:
            syllable: 要检查的音节
            rules: 音素规则字典
            entries: 所有条目列表
            current_entry: 当前正在处理的条目（排除在检查之外）
            
        返回:
            bool: 如果音素已在其他地方被覆盖则返回True，否则返回False
        """
        if syllable == 'R':
            return True  # R可以安全删除，因为它只是休止符
            
        if syllable not in rules:
            return True  # 未知音节可以安全删除
            
        # 获取音节的音素信息
        parts = rules[syllable]
        consonant = parts['consonant']
        onset = parts['onset']
        coda = parts['coda']
        transitions = parts['transition']
        
        # 初始化覆盖标志
        consonant_covered = not consonant  # 如果没有辅音，则视为已覆盖
        onset_covered = not onset  # 如果没有起始音，则视为已覆盖
        coda_covered = not coda  # 如果没有尾音，则视为已覆盖
        transitions_covered = all(not t[0] for t in transitions)  # 如果没有过渡音，则视为已覆盖
        
        # 检查每个条目中的音素覆盖情况
        for entry in entries:
            if entry == current_entry:
                continue  # 跳过当前正在处理的条目
                
            entry_parts = entry.split('_')
            for idx, part in enumerate(entry_parts):
                if part == syllable:
                    # 如果在其他条目中找到相同的音节，则所有音素都被覆盖
                    return True
                    
                if part not in rules:
                    continue
                    
                part_rules = rules[part]
                
                # 检查辅音覆盖
                if part_rules['consonant'] == consonant and consonant:
                    consonant_covered = True
                    
                # 检查起始音覆盖
                if part_rules['onset'] == onset and onset:
                    onset_covered = True
                    
                # 检查尾音覆盖
                if part_rules['coda'] == coda and coda:
                    coda_covered = True
                    
                # 检查过渡音覆盖
                for i, trans in enumerate(transitions):
                    if i < len(part_rules['transition']) and trans[0] and part_rules['transition'][i][0] == trans[0]:
                        transitions_covered = True
                        break
                        
                # 检查尾音-辅音组合覆盖
                if idx < len(entry_parts) - 1 and entry_parts[idx+1] in rules:
                    next_part = entry_parts[idx+1]
                    next_rules = rules[next_part]
                    if part_rules['coda'] == coda and next_rules['consonant'] == consonant and coda and consonant:
                        # 找到了相同的尾音-辅音组合，标记为已覆盖，但不立即返回
                        # 因为我们需要确保所有音素都被覆盖
                        consonant_covered = True
                        coda_covered = True
        
        # 检查所有音素是否都被覆盖
        all_covered = consonant_covered and onset_covered and coda_covered and transitions_covered
        
        # 输出调试信息
        if not all_covered:
            print(f"音节 '{syllable}' 不能安全删除，因为以下音素未被覆盖："
                  f"辅音({consonant_covered}), 起始音({onset_covered}), "
                  f"尾音({coda_covered}), 过渡音({transitions_covered})")
        else:
            print(f"音节 '{syllable}' 可以安全删除，所有音素已在其他条目中被覆盖")
            
        return all_covered
    
    @staticmethod
    def _merge_standard_entries(entries, max_syllables, rules):
        """合并标准生成的条目
        标准条目没有特殊的模式要求，可以简单地按最大音节数合并。
        现在增加了音素检查功能，确保删除单个音节不会导致音素覆盖缺失。
        """
        merged = []
        current = []
        
        # 创建一个副本用于音素覆盖检查
        all_entries = list(entries)
        
        for entry in sorted(entries, key=natural_sort_key):
            parts = entry.split('_')
            
            # 如果当前条目为空，或者添加新的部分不会超出最大音节数，则直接添加
            if not current or len(current) + len(parts) <= max_syllables:
                current.extend(parts)
            else:
                # 检查是否可以通过删除某些音节来腾出空间
                # 只考虑删除单音节条目，因为它们更容易被其他条目覆盖
                if len(parts) == 1 and len(current) == max_syllables:
                    # 检查新音节是否可以替换当前条目中的某个音节
                    for i, syl in enumerate(current):
                        if RuleParser._is_phoneme_covered_elsewhere(syl, rules, all_entries, '_'.join(current)):
                            # 可以安全删除这个音节
                            current[i] = parts[0]  # 替换为新音节
                            break
                    else:
                        # 没有找到可以安全删除的音节，保存当前条目并开始新条目
                        if current:
                            merged.append('_'.join(current))
                        current = parts
                else:
                    # 多音节条目或当前条目未满，保存当前条目并开始新条目
                    if current:
                        merged.append('_'.join(current))
                    current = parts
                
        # 添加最后一个条目
        if current:
            merged.append('_'.join(current))
            
        return merged
        
    @staticmethod
    def generate_standard_entries(rules, syllables, max_syllables, settings, initial_covered_phonemes=None):
        # 初始化覆盖率跟踪
        if initial_covered_phonemes is None:
            initial_covered_phonemes = set()
        
        coverage = {
            'start': set(),      # 开头整音覆盖
            'middle': set(),     # 中间整音覆盖
            'transition': defaultdict(set),  # transition覆盖
            'end_coda': set(),   # 结尾coda覆盖
            'coda_onset': set(), # VCV模式专用
            'coda_consonant': set()  # VCCV-CVVC模式
        }
        
        # 合并初始覆盖率
        for item in initial_covered_phonemes:
            if isinstance(item, tuple):
                if len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], str):
                    coverage['coda_onset'].add(item)  # VCV模式组合
            elif any(item == parts.get('coda') for parts in rules.values()):
                coverage['end_coda'].add(item)
            else:
                coverage['start'].add(item)
                coverage['middle'].add(item)

        # 收集关键组合
        critical_combinations = {
            'start': set(),
            'middle': set(),
            'transition': set(),
            'end_coda': set(),
            'coda_onset': set(),  # VCV模式新增
            'coda_consonant': set()  # VCCV-CVVC模式
        }

        # 1. 开头整音覆盖
        for syl in syllables:
            if syl not in coverage['start']:
                critical_combinations['start'].add(syl)
        
        # 2. 中间整音覆盖
        for syl in syllables:
            if syl not in coverage['middle']:
                critical_combinations['middle'].add(syl)
        
        # 3. Transition元素收集
        for syl, parts in rules.items():
            for idx, trans_group in enumerate(parts.get('transition', [])):
                if trans_group and trans_group[0] and idx not in coverage['transition'][syl]:
                    critical_combinations['transition'].add((syl, idx))
        
        # 4. 结尾coda覆盖
        all_codas = {parts.get('coda') for parts in rules.values() if parts.get('coda')}
        missing_end_codas = all_codas - coverage['end_coda']
        syllables_with_coda = defaultdict(list)
        for syl, parts in rules.items():
            if parts.get('coda'):
                syllables_with_coda[parts['coda']].append(syl)
        
        # 5. 模式特定组合收集
        mode = settings.get('mode', 'vccv-cvvc')
        remaining_combos = defaultdict(set)
        
        if mode == "vccv-cvvc":
            # VCCV-CVVC: coda -> consonant
            all_consonants = {parts.get('consonant') for parts in rules.values() if parts.get('consonant')}
            for coda in all_codas:
                for consonant in all_consonants:
                    if (coda, consonant) not in coverage['coda_consonant']:
                        remaining_combos[coda].add(consonant)
        elif mode == "vcv":
            # VCV: coda -> onset
            all_onsets = {parts.get('onset') for parts in rules.values() if parts.get('onset')}
            for coda in all_codas:
                for onset in all_onsets:
                    if (coda, onset) not in coverage['coda_onset']:
                        remaining_combos[coda].add(onset)
        elif mode == "cvc":
            # CVC: onset -> consonant
            all_onsets = {parts.get('onset') for parts in rules.values() if parts.get('onset')}
            all_consonants = {parts.get('consonant') for parts in rules.values() if parts.get('consonant')}
            for onset in all_onsets:
                for consonant in all_consonants:
                    if (onset, consonant) not in coverage['onset_consonant']:
                        remaining_combos[onset].add(consonant)

        # 生成基础条目
        standard_entries = set()
        
        # 1. 添加开头整音
        for syl in critical_combinations['start']:
            standard_entries.add(syl)
            coverage['start'].add(syl)
            coverage['middle'].add(syl)
        
        # 2. 添加中间整音
        for syl in critical_combinations['middle']:
            standard_entries.add(syl)
            coverage['middle'].add(syl)
        
        # 3. 添加transition条目
        for syl, idx in critical_combinations['transition']:
            standard_entries.add(syl)
            if syl not in coverage['transition']:
                coverage['transition'][syl] = set()
            coverage['transition'][syl].add(idx)
        
        # 4. 添加结尾coda R条目
        for coda in missing_end_codas:
            if syllables_with_coda[coda]:
                syl_with_coda = syllables_with_coda[coda][0]
                standard_entries.add(f"{syl_with_coda}_R")
                coverage['end_coda'].add(coda)
        
        # 5. 添加模式特定组合
        if mode == "vccv-cvvc":
            # 处理coda-consonant组合
            for coda, consonants in remaining_combos.items():
                for consonant in consonants:
                    coda_syl = next((s for s, p in rules.items() if p.get('coda') == coda), None)
                    cons_syl = next((s for s, p in rules.items() if p.get('consonant') == consonant), None)
                    if coda_syl and cons_syl:
                        entry = f"{coda_syl}_{cons_syl}"
                        standard_entries.add(entry)
                        coverage['coda_consonant'].add((coda, consonant))
        elif mode == "vcv":
            # 处理coda-onset组合（VCV模式核心修改）
            for coda, onsets in remaining_combos.items():
                for onset in onsets:
                    coda_syl = next((s for s, p in rules.items() if p.get('coda') == coda), None)
                    onset_syl = next((s for s, p in rules.items() if p.get('onset') == onset), None)
                    if coda_syl and onset_syl:
                        entry = f"{coda_syl}_{onset_syl}"
                        standard_entries.add(entry)
                        coverage['coda_onset'].add((coda, onset))
        elif mode == "cvc":
            # 处理onset-consonant组合
            for onset, consonants in remaining_combos.items():
                for consonant in consonants:
                    onset_syl = next((s for s, p in rules.items() if p.get('onset') == onset), None)
                    cons_syl = next((s for s, p in rules.items() if p.get('consonant') == consonant), None)
                    if onset_syl and cons_syl:
                        entry = f"{onset_syl}_{cons_syl}"
                        standard_entries.add(entry)
                        coverage['onset_consonant'].add((onset, consonant))
        
        # 确保所有音节至少出现一次
        covered_syllables = set()
        for entry in standard_entries:
            covered_syllables.update(entry.split('_'))
        
        missing_syllables = set(syllables) - covered_syllables
        standard_entries.update(missing_syllables)
        
        # 确保R后面的音节作为开头整音出现
        for entry in list(standard_entries):
            parts = entry.split('_')
            for i in range(1, len(parts)):
                if parts[i-1] == 'R' and parts[i] in rules:
                    if parts[i] not in coverage['start']:
                        standard_entries.add(parts[i])
                        coverage['start'].add(parts[i])
        
        return list(standard_entries)

    @staticmethod
    def _merge_standard_entries(entries, max_syllables, rules):
        """合并标准生成的条目，保持音素覆盖完整性"""
        merged = []
        current = []
        
        # 按自然排序确保一致性
        sorted_entries = sorted(entries, key=lambda x: (len(x.split('_')), x))
        
        for entry in sorted_entries:
            parts = entry.split('_')
            # 如果当前条目为空，或者添加新的部分不会超出最大音节数
            if not current or len(current) + len(parts) <= max_syllables:
                current.extend(parts)
            else:
                # 检查是否可以通过删除某些单音节条目腾出空间
                if len(parts) == 1 and len(current) == max_syllables:
                    for i, syl in enumerate(current):
                        if syl not in coverage['start'] and syl not in coverage['middle']:
                            # 替换可删除的单音节
                            current[i:i+1] = parts
                            break
                    else:
                        # 无法替换则新建条目
                        merged.append('_'.join(current))
                        current = parts
                else:
                    # 无法添加则新建条目
                    merged.append('_'.join(current))
                    current = parts
        
        # 添加最后一个条目
        if current:
            merged.append('_'.join(current))
        
        return merged

    @staticmethod
    def generate_recording_list(rules, settings):
        """生成录音列表，结合强制生成模式和标准覆盖生成"""
        max_syllables = settings['max_syllables_per_sentence']
        generation_mode = settings.get('generation_mode', 'none')
        auto_merge = settings.get('auto_merge', True)
        syllables = list(rules.keys())
        
        # 初始化覆盖率跟踪
        coverage = {
            'start': set(),
            'middle': set(),
            'transition': defaultdict(set),
            'end_coda': set(),
            'coda_onset': set(),
            'coda_consonant': set(),
            'onset_consonant': set()
        }
        
        # 存储强制生成条目
        forced_entries = []
        
        # 执行强制生成模式
        if generation_mode in ['repeat', 'interval', 'sequence']:
            mode_handlers = {
                'repeat': lambda: RuleParser.handle_repeat(rules, syllables, forced_entries, coverage),
                'interval': lambda: RuleParser.handle_interval(rules, syllables, forced_entries, coverage, max_syllables),
                'sequence': lambda: RuleParser.handle_sequence(rules, syllables, forced_entries, coverage, max_syllables)
            }
            mode_handlers[generation_mode]()
        
        # 计算初始覆盖率
        initial_covered_phonemes = set()
        for entry in forced_entries:
            parts = entry.split('_')
            for i, part in enumerate(parts):
                if part in rules:
                    rule = rules[part]
                    if rule.get('onset'):
                        initial_covered_phonemes.add(rule['onset'])
                    if rule.get('consonant'):
                        initial_covered_phonemes.add(rule['consonant'])
                    if rule.get('coda'):
                        initial_covered_phonemes.add(rule['coda'])
                    for idx, trans in enumerate(rule.get('transition', [])):
                        if trans:
                            initial_covered_phonemes.add(trans.strip('"'))
                    # 处理VC组合
                    if i < len(parts)-1 and parts[i+1] in rules:
                        next_rule = rules[parts[i+1]]
                        if rule.get('coda') and next_rule.get('consonant'):
                            initial_covered_phonemes.add((rule['coda'], next_rule['consonant']))
        
        # 生成标准条目
        standard_entries = RuleParser.generate_standard_entries(rules, syllables, max_syllables, initial_covered_phonemes)
        
        # 应用合并策略
        if auto_merge and generation_mode == 'none':
            standard_entries = RuleParser._merge_standard_entries(standard_entries, max_syllables, rules)
        
        # 合并所有条目
        final_entries = forced_entries + standard_entries
        
        # 确保没有重复条目
        unique_entries = []
        seen = set()
        for entry in final_entries:
            if entry not in seen:
                seen.add(entry)
                unique_entries.append(entry)
        
        return unique_entries

    @staticmethod 
    def _build_coda_consonant_graph(rules): 
        """构建音素依赖关系图 - 优化版本"""
        valid_consonants = set()
        valid_codas = set()
        coda_to_syllables = defaultdict(list)
        consonant_to_syllables = defaultdict(list)

        for syl, parts in rules.items():
            if parts['consonant']:
                valid_consonants.add(parts['consonant'])
                consonant_to_syllables[parts['consonant']].append(syl)
            if parts['coda']:
                valid_codas.add(parts['coda'])
                coda_to_syllables[parts['coda']].append(syl)

        graph = defaultdict(dict)
        for coda in valid_codas:
            for consonant in valid_consonants:
                graph[coda][consonant] = len(coda_to_syllables[coda]) * len(consonant_to_syllables[consonant])
        return graph

    @staticmethod
    def _cover_critical_combinations(standard_entries, rules, syllables, coverage):
        """优先覆盖关键音素组合"""
        for syl in syllables:
            if syl not in coverage['start']:
                standard_entries.add(syl)
            if syl not in coverage['middle']:
                standard_entries.add(syl)

        all_codas = {parts['coda'] for parts in rules.values() if parts['coda']}
        missing_end_codas = all_codas - coverage['end_coda']
        syllables_with_coda = defaultdict(list)
        
        for syl, parts in rules.items():
            if parts['coda']:
                syllables_with_coda[parts['coda']].append(syl)

        for coda in missing_end_codas:
            if syllables_with_coda[coda]:
                standard_entries.add(syllables_with_coda[coda][0])

    @staticmethod
    def _get_remaining_combinations(coverage, coda_consonant_graph):
        """获取未覆盖的音素组合 - 优化版本"""
        remaining = defaultdict(set)
        covered_pairs = coverage['coda_consonant']

        for coda, consonants_dict in coda_consonant_graph.items():
            if not coda:
                continue
                
            all_consonants = set(consonants_dict.keys())
            covered_consonants = set(cons for c, cons in covered_pairs if c == coda)
            uncovered_consonants = all_consonants - covered_consonants

            if uncovered_consonants:
                remaining[coda] = uncovered_consonants
                print(f"尾音 '{coda}' 还有 {len(uncovered_consonants)} 个未覆盖的辅音组合需要生成")

        total_uncovered = sum(len(consonants) for consonants in remaining.values())
        if total_uncovered > 0:
            print(f"总计有 {total_uncovered} 个尾音-辅音组合需要由标准生成算法覆盖")
        else:
            print("所有尾音-辅音组合已被强制生成模式完全覆盖")
        
        return remaining

    @staticmethod
    def _greedy_coverage(standard_entries, remaining_combos, rules):
        """贪心算法实现组合覆盖"""
        while remaining_combos:
            best_coda = None
            best_consonant = None
            max_score = -1

            for coda, consonants in remaining_combos.items():
                for consonant in consonants:
                    score = RuleParser._calculate_coverage_score(coda, consonant, 
                        defaultdict(list, {c: [s for s in rules if rules[s]['coda'] == c]}),
                        defaultdict(list, {c: [s for s in rules if rules[s]['consonant'] == c]}))
                    if score > max_score:
                        max_score = score
                        best_coda = coda
                        best_consonant = consonant

            if best_coda and best_consonant:
                entry = f"{random.choice([s for s in rules if rules[s]['coda'] == best_coda])}_{random.choice([s for s in rules if rules[s]['consonant'] == best_consonant])}"
                standard_entries.add(entry)
                remaining_combos[best_coda].discard(best_consonant)
                if not remaining_combos[best_coda]:
                    del remaining_combos[best_coda]

    @staticmethod
    def _add_missing_syllables(standard_entries, syllables, coverage, rules, settings):
        """补充未覆盖的独立音节"""
        missing_start = [s for s in syllables if s not in coverage['start']]
        missing_middle = [s for s in syllables if s not in coverage['middle']]
        
        if missing_start:
            standard_entries.update(random.sample(missing_start, min(3, len(missing_start))))
        if missing_middle:
            standard_entries.update(random.sample(missing_middle, min(3, len(missing_middle))))
    
    @staticmethod 
    def _build_coda_consonant_graph(rules): 
        """构建音素依赖关系图 - 优化版本"""
        # 预先提取所有有效的辅音和尾音，避免重复计算
        valid_consonants = set()
        valid_codas = set()
        coda_to_syllables = defaultdict(list)
        consonant_to_syllables = defaultdict(list)
        
        # 第一次遍历：收集所有有效的辅音和尾音
        for syl, parts in rules.items():
            if parts['consonant']:
                valid_consonants.add(parts['consonant'])
                consonant_to_syllables[parts['consonant']].append(syl)
            if parts['coda']:
                valid_codas.add(parts['coda'])
                coda_to_syllables[parts['coda']].append(syl)
        
        # 构建图 - 直接使用已知的有效辅音和尾音
        graph = defaultdict(dict)
        for coda in valid_codas:
            for consonant in valid_consonants:
                # 只在图中存储有效的组合，并记录组合的频率
                graph[coda][consonant] = len(coda_to_syllables[coda]) * len(consonant_to_syllables[consonant])
        
        return graph
    
    @staticmethod
    def _cover_critical_combinations(standard_entries, rules, syllables, coverage):
        """优先覆盖关键音素组合"""
        # 1. 检查并添加未覆盖的单个音节 (起始和中间)
        for syl in syllables:
            if syl not in coverage['start']:
                standard_entries.add(syl) # 优先添加单个音节以覆盖起始
            if syl not in coverage['middle']:
                standard_entries.add(syl)
        
        # 2. 检查未覆盖的结尾 Coda
        all_codas = {parts['coda'] for parts in rules.values() if parts['coda']}
        missing_end_codas = all_codas - coverage['end_coda']
        
        syllables_with_coda = defaultdict(list)
        for syl, parts in rules.items():
            if parts['coda']:
                syllables_with_coda[parts['coda']].append(syl)
                
        for coda in missing_end_codas:
            # 需要生成以带这个 coda 的音节结尾的条目
            if syllables_with_coda[coda]:
                syl_with_coda = syllables_with_coda[coda][0]
                standard_entries.add(syl_with_coda) # 添加单个音节作为结尾
    
    @staticmethod
    def _get_remaining_combinations(coverage, coda_consonant_graph):
        """获取未覆盖的音素组合 - 优化版本
        考虑强制生成模式已覆盖的音素组合，确保标准生成只填补未覆盖的部分
        """
        remaining = defaultdict(set)
        covered_pairs = coverage['coda_consonant']
        
        # 使用集合操作更高效地找出未覆盖的组合
        for coda, consonants_dict in coda_consonant_graph.items():
            # 只处理有效的尾音
            if not coda:
                continue
                
            # 获取当前尾音可以连接的所有辅音
            all_consonants = set(consonants_dict.keys())
            
            # 找出已覆盖的辅音 - 这里考虑了强制生成模式已覆盖的组合
            covered_consonants = set(cons for c, cons in covered_pairs if c == coda)
            
            # 计算未覆盖的辅音（差集操作）
            uncovered_consonants = all_consonants - covered_consonants
            
            # 只添加有效的未覆盖组合
            if uncovered_consonants:
                remaining[coda] = uncovered_consonants
                print(f"尾音 '{coda}' 还有 {len(uncovered_consonants)} 个未覆盖的辅音组合需要生成")
        
        # 输出总体覆盖情况
        total_uncovered = sum(len(consonants) for consonants in remaining.values())
        if total_uncovered > 0:
            print(f"总计有 {total_uncovered} 个尾音-辅音组合需要由标准生成算法覆盖")
        else:
            print("所有尾音-辅音组合已被强制生成模式完全覆盖")
            
        return remaining
    
    @staticmethod
    def _calculate_coverage_score(coda, consonant, coda_syllables, consonant_syllables):
        """计算组合的覆盖价值 - 优化版本"""
        # 直接使用预计算的音节列表长度，避免遍历整个规则集
        return len(coda_syllables.get(coda, [])) * len(consonant_syllables.get(consonant, []))
    
    @staticmethod
    def _find_best_syllable(coda_syllables, consonant_syllables, coda=None, consonant=None):
        """找到最适合的音节 - 优化版本"""
        if coda and coda in coda_syllables and coda_syllables[coda]:
            return coda_syllables[coda][0]
        elif consonant and consonant in consonant_syllables and consonant_syllables[consonant]:
            return consonant_syllables[consonant][0]
        return None
    
    @staticmethod
    def _update_remaining_combos(remaining_combos, combo_to_remove):
        """更新剩余组合列表 - 优化版本"""
        # 由于我们已经重构了贪心算法，这个方法实际上不再需要
        # 保留此方法是为了兼容性，但它现在只是一个空操作
        return remaining_combos
    
    @staticmethod
    def _greedy_coverage(entries, remaining_combos, rules):
        """贪心算法实现最优组合选择 - 优化版本""" 
        # 预计算所有组合的分数，避免重复计算
        combo_scores = {}
        coda_syllables = defaultdict(list)
        consonant_syllables = defaultdict(list)
        
        # 预处理：建立音素到音节的映射
        for syl, parts in rules.items():
            if parts['coda']:
                coda_syllables[parts['coda']].append(syl)
            if parts['consonant']:
                consonant_syllables[parts['consonant']].append(syl)
        
        # 预计算所有组合的分数
        for coda, consonants in remaining_combos.items():
            for cons in consonants:
                # 使用音节数量作为分数，避免遍历整个规则集
                score = len(coda_syllables[coda]) * len(consonant_syllables[cons])
                combo_scores[(coda, cons)] = score
        
        # 按分数降序排序所有组合
        sorted_combos = sorted(
            [(coda, cons) for coda, consonants in remaining_combos.items() for cons in consonants],
            key=lambda x: combo_scores.get(x, 0),
            reverse=True
        )
        
        # 贪心选择最高分数的组合
        covered_combos = set()
        for coda, cons in sorted_combos:
            # 如果组合已被覆盖，跳过
            if (coda, cons) in covered_combos:
                continue
                
            # 找到最适合的音节
            syl1 = coda_syllables[coda][0] if coda_syllables[coda] else None
            syl2 = consonant_syllables[cons][0] if consonant_syllables[cons] else None
            
            if syl1 and syl2:
                entries.add(f"{syl1}_{syl2}")
                covered_combos.add((coda, cons))
                
                # 更新已覆盖的组合集合
                # 这里可以添加额外的逻辑来标记那些通过这个条目间接覆盖的组合
        
        # 不再需要循环更新remaining_combos，因为我们已经处理了所有组合
    
    @staticmethod
    def _add_missing_syllables(standard_entries, syllables, coverage, rules, settings):
        """确保所有音节至少出现一次，并确保所有音素类型都被覆盖 - 增强版本"""
        # 收集已经在条目中出现的所有音节
        covered_syllables = set()
        for entry in standard_entries:
            covered_syllables.update(entry.split('_'))
        
        # 使用集合差集操作找出未覆盖的音节
        missing_syllables = set(syllables) - covered_syllables
        
        # 将未覆盖的音节添加到结果中
        standard_entries.update(missing_syllables)
        print(f"添加了 {len(missing_syllables)} 个未覆盖的音节")
        
        # --- 新增：确保所有音素类型都被覆盖 ---
        # 1. 检查所有开头整音（前面是空的整音）
        for syl, rule in rules.items():
            if rule['onset'] and syl not in coverage['start']:
                standard_entries.add(syl)
                coverage['start'].add(syl)
                print(f"添加开头整音: {syl}")
        
        # 2. 检查所有transition元素（介音，双元音等）
        for syl, rule in rules.items():
            for idx, trans_group in enumerate(rule['transition']):
                if trans_group and trans_group[0] and idx not in coverage['transition'][syl]:
                    standard_entries.add(syl)
                    if syl not in coverage['transition']:
                        coverage['transition'][syl] = set()
                    coverage['transition'][syl].add(idx)
                    print(f"添加transition元素: {syl} (索引 {idx})")
        
        # 3. 检查所有coda到R的连接
        for syl, rule in rules.items():
            if rule['coda']:
                # 检查是否已经有这个coda到R的连接
                coda_r_exists = False
                for e in standard_entries:
                    if '_' in e and e.endswith('_R'):
                        prev_syl = e.split('_')[-2]
                        if prev_syl in rules and rules[prev_syl]['coda'] == rule['coda']:
                            coda_r_exists = True
                            break
                
                if not coda_r_exists:
                    standard_entries.add(f"{syl}_R")
                    print(f"添加coda到R连接: {syl}_R")
        
        # 4. 确保前面有R的拼音正确标记为开头整音
        for entry in list(standard_entries):
            parts = entry.split('_')
            for i in range(1, len(parts)):
                if parts[i-1] == 'R' and parts[i] in rules:
                    # 确保这个音节也作为开头整音出现
                    if parts[i] not in coverage['start']:
                        standard_entries.add(parts[i])
                        coverage['start'].add(parts[i])
                        print(f"添加R后的开头整音: {parts[i]}")
        
        # 5. 确保所有音素连接组合都被覆盖（根据当前模式）
        mode = settings.get('mode', 'vccv-cvvc')
        all_codas = {parts['coda'] for parts in rules.values() if parts['coda']}
        all_consonants = {parts['consonant'] for parts in rules.values() if parts['consonant']}
        all_onsets = {parts['onset'] for parts in rules.values() if parts['onset']}
        
        if mode == "cvc":
            # CVC模式：确保所有整音(onset)到辅音(consonant)的连接都被覆盖
            for onset in all_onsets:
                for consonant in all_consonants:
                    # 检查这个onset-consonant组合是否已经被覆盖
                    # 注意：CVC模式下我们仍然使用coda_consonant集合来跟踪，但实际存储的是onset-consonant对
                    if (onset, consonant) not in coverage.get('onset_consonant', set()):
                        # 找到具有这个onset的音节
                        onset_syllable = None
                        for syl, parts in rules.items():
                            if parts['onset'] == onset:
                                onset_syllable = syl
                                break
                        
                        # 找到具有这个consonant的音节
                        consonant_syllable = None
                        for syl, parts in rules.items():
                            if parts['consonant'] == consonant:
                                consonant_syllable = syl
                                break
                        
                        if onset_syllable and consonant_syllable:
                            entry = f"{onset_syllable}_{consonant_syllable}"
                            standard_entries.add(entry)
                            if 'onset_consonant' not in coverage:
                                coverage['onset_consonant'] = set()
                            coverage['onset_consonant'].add((onset, consonant))
                            print(f"添加onset-consonant连接: {entry} ({onset} {consonant})")
        else:
            # VCCV-CVVC和VCV模式：确保所有coda到consonant的连接都被覆盖
            for coda in all_codas:
                for consonant in all_consonants:
                    if (coda, consonant) not in coverage['coda_consonant']:
                        # 找到具有这个coda的音节
                        coda_syllable = None
                        for syl, parts in rules.items():
                            if parts['coda'] == coda:
                                coda_syllable = syl
                                break
                        
                        # 找到具有这个consonant的音节
                        consonant_syllable = None
                        for syl, parts in rules.items():
                            if parts['consonant'] == consonant:
                                consonant_syllable = syl
                                break
                        
                        if coda_syllable and consonant_syllable:
                            entry = f"{coda_syllable}_{consonant_syllable}"
                            standard_entries.add(entry)
                            coverage['coda_consonant'].add((coda, consonant))
                            print(f"添加coda-consonant连接: {entry} ({coda} {consonant})")
        
        return standard_entries

# -------------------------- GUI 相关类和函数 --------------------------

    @staticmethod
    def generate(rules, recording_list, params, max_alternatives, mode, generation_mode):
        entries = []
        beat_ms = 60000 / params['bpm']
        leading_silence = params['leading_silence']
        alias_count = defaultdict(int)

        for line in recording_list:
            # 保持使用 '_' 分割录音列表条目
            aliases = line.split('_')
            # 文件名保持基于原始 line 生成
            filename = f"{line}.wav"
            total_duration = beat_ms * len(aliases)
            current_time = leading_silence

            for idx, alias in enumerate(aliases):
                alias = alias.strip()

                if alias == 'R':
                    # 在 interval 模式下，R 只是占位符，不生成 OTO 条目，只推进时间
                    if generation_mode == "interval":
                        current_time += beat_ms
                        continue
                    # 在其他模式下，如果出现 R，则生成 R 的 OTO 条目
                    alias_list = ['R']
                else:
                    # 处理普通音节
                    if alias not in rules:
                        print(f"Warning: Alias '{alias}' not defined in rules, skipped in OTO generation for line '{line}'")
                        # 如果规则中没有定义，可以选择跳过这个音节或者采取其他处理
                        # 这里选择跳过当前音节的处理，但时间仍然可能需要推进（取决于后续逻辑）
                        # 为了避免时间计算错误，如果规则未定义，最好也跳过后续的 OTO 生成和时间推进
                        # 或者，如果希望即使规则未定义也占一个节拍时间，可以在这里 current_time += beat_ms
                        # 当前选择：打印警告并完全跳过此未知音节的 OTO 生成
                        continue # 跳到下一个 alias

                    parts = rules[alias]
                    onset = parts['onset'] # 获取 onset (声母或零声母对应的音素)
                    coda = parts['coda']   # 获取 coda (韵尾音素，可能为空)

                    # 确定基础别名：行首或 R 之后是 "- onset"，否则是 "onset"
                    alias_name = f"- {onset}" if idx == 0 or (idx > 0 and aliases[idx-1] == 'R') else onset
                    alias_list = [alias_name] # 初始化当前音节要生成的 OTO 别名列表

                    # 添加尾部 coda 支持
                    if idx == len(aliases) - 1 and coda:
                        alias_list.append(coda)

                    # --- 处理 VCCV/VCV/CVC 过渡音 ---
                    # 检查是否有下一个音节，并且下一个音节不是 'R'
                    if idx < len(aliases) - 1 and aliases[idx+1] != 'R':
                        next_alias = aliases[idx+1]
                        if next_alias in rules:
                            next_parts = rules[next_alias]
                            # 获取下一个音节的辅音，如果是'无辅音'则为空字符串
                            next_consonant = next_parts['consonant'] if next_parts['consonant'] != '无辅音' else ''
                            next_onset = next_parts['onset'] # 获取下一个音节的 onset

                            transition = "" # 初始化过渡音素字符串
                            
                            # 根据不同模式生成过渡音素
                            if mode == "vccv-cvvc":
                                # VCCV 模式：韵尾 + 下一个辅音
                                if coda and next_consonant: # 只有当前音节有韵尾且下一个音节有辅音时才生成
                                    transition = f"{coda} {next_consonant}"
                            elif mode == "vcv":
                                # VCV 模式：韵尾 + 下一个 onset
                                if coda and next_onset: # 只有当前音节有韵尾且下一个音节有整音时才生成
                                    transition = f"{coda} {next_onset}"
                            elif mode == "cvc":
                                # CVC 模式：当前 onset + 下一个辅音
                                # 注意：CVC模式下不需要检查coda，而是使用onset
                                if onset and next_consonant: # 只有当前音节有整音且下一个音节有辅音时才生成
                                    transition = f"{onset} {next_consonant}"
                            # 如果生成了有效的过渡音素（去除空格后不为空）
                            if transition.strip():
                                alias_list.append(transition.strip()) # 添加到别名列表

                    # --- 处理间隔式 (Interval) 下的 syl_R 情况 ---
                    # 检查是否有下一个音节，下一个音节是 'R'，并且当前音节有韵尾 (coda)
                    if idx < len(aliases) - 1 and aliases[idx+1] == 'R' and coda:
                        # 添加 "韵尾 R" 格式的别名
                        alias_list.append(f"{coda} R")
                    # 若是最后一个音节且有 coda，则添加 coda 作为独立别名
                    if idx == len(aliases) - 1 and coda:
                        alias_list.append(coda)

                # --- 为当前音节生成所有 OTO 条目 ---
                for alias_item in alias_list:
                    count = alias_count[alias_item]

                    # 处理别名重复序号（使用 #）
                    current_alias_for_entry = alias_item # 默认使用原始别名
                    if count > 0: # 如果不是第一次出现
                        if max_alternatives == 1: # 如果只允许1个，跳过后续重复
                            continue
                        elif count < max_alternatives: # 如果允许多个且未达上限
                            current_alias_for_entry = f"{alias_item}#{count}" # 使用 # 作为分隔符
                        else: # 达到或超过上限，跳过
                            continue

                    alias_count[alias_item] += 1 # 增加此别名的计数

                    # 添加 OTO 条目到最终列表
                    entries.append({
                        'filename': filename,
                        'alias': current_alias_for_entry, # 使用处理后的别名
                        'offset': current_time,          # 当前时间点作为偏移量
                        'consonant': beat_ms * 0.4,      # 辅音部分时长（示例值）
                        'cutoff': -(total_duration - (current_time - leading_silence + beat_ms)), # 截止点（负数，相对文件末尾）
                        'preutterance': beat_ms * 0.2,   # 先行发声（示例值）
                        'overlap': beat_ms * 0.1         # 重叠（示例值）
                    })

                # --- 时间推进 ---
                # 只有处理普通音节 (非 R) 时才推进时间
                # 或者在非 interval 模式下遇到 R 时也推进时间
                if alias != 'R' or (alias == 'R' and generation_mode != "interval"):
                     current_time += beat_ms


        return entries
    
class OTOGenerator:
    @staticmethod
    def generate(rules, recording_list, params, max_alternatives, mode, generation_mode):
        entries = []
        beat_ms = 60000 / params['bpm']
        leading_silence = params['leading_silence']
        alias_count = defaultdict(int)
        
        # 跟踪已处理的音素组合，避免重复
        processed_combinations = set()

        for line in recording_list:
            aliases = line.split('_')
            filename = f"{line}.wav"
            total_duration = beat_ms * len(aliases)
            current_time = leading_silence

            for idx, alias in enumerate(aliases):
                alias = alias.strip()
                
                if alias == 'R':
                    # 在间隔模式下，R只作为间隔，不生成OTO条目
                    if generation_mode == "interval":
                        current_time += beat_ms
                        continue
                    # 在其他模式下，R作为独立音素
                    alias_list = ['R']
                else:
                    if alias not in rules:
                        print(f"警告: 别名 '{alias}' 未在规则中定义，已跳过。")
                        continue
                        
                    parts = rules[alias]
                    consonant = parts['consonant']
                    onset = parts['onset']
                    coda = parts['coda']
                    transitions = parts['transition']
                    
                    # 1. 处理开头整音或普通整音
                    # 如果是句首或前一个是R，使用带前导符号的整音
                    alias_name = f"- {onset}" if idx == 0 or aliases[idx-1] == 'R' else onset
                    alias_list = [alias_name]
                    
                    # 2. 处理transition元素（介音，双元音等）
                    # 遍历所有transition组，确保每个有效的transition都被添加
                    for i, trans_group in enumerate(transitions):
                        for trans in trans_group:
                            if trans and trans.strip():
                                # 添加每个transition元素作为独立的别名
                                alias_list.append(trans.strip())
                                # 记录已处理的transition元素
                                processed_combinations.add(("transition", alias, i, trans.strip()))
                    
                    # 3. 处理coda与下一个发音的连接
                    if idx < len(aliases)-1 and aliases[idx+1] != 'R':
                        next_alias = aliases[idx+1]
                        if next_alias in rules:
                            next_parts = rules[next_alias]
                            next_consonant = next_parts['consonant']
                            next_onset = next_parts['onset']
                            
                            # 根据不同模式生成连接
                            # 初始化transition变量，确保在所有情况下都有定义
                            transition = ""
                            
                            # 只有当前音节有coda或下一个音节有consonant时才生成连接
                            if coda or next_consonant:
                                if mode == "vccv-cvvc":
                                    # VCCV-CVVC模式: 当前音节的coda到下一个音节的辅音
                                    if coda and next_consonant:
                                        # 使用空格作为连接符号，而不是'-'
                                        transition = f"{coda} {next_consonant}"
                                        # 记录已处理的coda-consonant组合
                                        processed_combinations.add((coda, next_consonant))
                                elif mode == "vcv":
                                    # VCV模式: 当前音节的coda到下一个音节的整音
                                    if coda:
                                        transition = f"{coda} {next_onset}"
                                        processed_combinations.add((coda, "onset", next_onset))
                                elif mode == "cvc":
                                    # CVC模式: 当前音节的整音到下一个音节的辅音
                                    if next_consonant:
                                        transition = f"{onset} {next_consonant}"
                                        processed_combinations.add((onset, next_consonant))
                                
                                # 如果生成了有效的过渡音素（去除空格后不为空）
                                if transition and transition.strip():
                                    alias_list.append(transition.strip())
                    
                    # 4. 处理coda到R的连接
                    if idx < len(aliases)-1 and aliases[idx+1] == 'R' and coda:
                        coda_r = f"{coda} R"
                        alias_list.append(coda_r)
                        processed_combinations.add((coda, "R"))
                    
                    # 5. 处理末尾音节的coda
                    if idx == len(aliases)-1 and coda:
                        # 添加独立的coda作为结尾
                        alias_list.append(coda)
                        processed_combinations.add(("end", coda))

                # 为当前音节的所有别名生成OTO条目
                for alias_item in alias_list:
                    # 处理别名重复
                    count = alias_count[alias_item]
                    alias_unique = alias_item  # 默认使用原始别名
                    
                    # 如果已经出现过且未超过最大重复数，添加序号
                    if 0 < count <= max_alternatives:
                        alias_unique = f"{alias_item}#{count}"  # 使用#作为分隔符
                    # 如果超过最大重复数，跳过
                    elif count > max_alternatives:
                        continue
                    
                    # 增加此别名的计数
                    alias_count[alias_item] += 1
                    
                    # 添加OTO条目
                    entries.append({
                        'filename': filename,
                        'alias': alias_unique,
                        'offset': current_time,
                        'consonant': beat_ms * 0.4,
                        'cutoff': -(total_duration - (current_time - leading_silence + beat_ms)),
                        'preutterance': beat_ms * 0.2,
                        'overlap': beat_ms * 0.1
                    })
                    
                # 时间推进 - 只在处理完一个完整音节后推进
                if alias != 'R' or (alias == 'R' and generation_mode != "interval"):
                    current_time += beat_ms

        # 检查是否有未覆盖的音素组合，并添加额外的OTO条目
        OTOGenerator._add_missing_combinations(entries, rules, processed_combinations, mode, alias_count, max_alternatives)
        
        return entries
    
    @staticmethod
    def _add_missing_combinations(entries, rules, processed_combinations, mode, alias_count, max_alternatives):
        """添加未被覆盖的音素组合"""
        # 收集所有有效的coda和consonant
        all_codas = set(parts['coda'] for parts in rules.values() if parts['coda'])
        all_consonants = set(parts['consonant'] for parts in rules.values() if parts['consonant'])
        all_onsets = set(parts['onset'] for parts in rules.values())
        
        # 根据模式检查未覆盖的组合
        if mode == "vccv-cvvc":
            # 检查所有coda-consonant组合
            for coda in all_codas:
                for consonant in all_consonants:
                    if (coda, consonant) not in processed_combinations:
                        # 找到具有这个coda的音节和具有这个consonant的音节
                        coda_syllable = next((syl for syl, parts in rules.items() if parts['coda'] == coda), None)
                        consonant_syllable = next((syl for syl, parts in rules.items() if parts['consonant'] == consonant), None)
                        
                        if coda_syllable and consonant_syllable:
                            # 创建一个虚拟文件名
                            filename = f"{coda_syllable}_{consonant_syllable}.wav"
                            # 创建别名
                            alias_item = f"{coda} {consonant}"
                            
                            # 处理别名重复
                            count = alias_count[alias_item]
                            alias_unique = alias_item
                            if 0 < count <= max_alternatives:
                                alias_unique = f"{alias_item}#{count}"
                            elif count > max_alternatives:
                                continue
                            
                            alias_count[alias_item] += 1
                            
                            # 添加OTO条目
                            entries.append({
                                'filename': filename,
                                'alias': alias_unique,
                                'offset': 200,  # 默认偏移
                                'consonant': 100,  # 默认值
                                'cutoff': -100,  # 默认值
                                'preutterance': 50,  # 默认值
                                'overlap': 25  # 默认值
                            })
                            
                            print(f"添加未覆盖的coda-consonant组合: {alias_item}")
        
        elif mode == "vcv":
            # 检查所有coda-onset组合
            for coda in all_codas:
                for onset in all_onsets:
                    if (coda, "onset", onset) not in processed_combinations:
                        # 找到具有这个coda的音节
                        coda_syllable = next((syl for syl, parts in rules.items() if parts['coda'] == coda), None)
                        onset_syllable = next((syl for syl, parts in rules.items() if parts['onset'] == onset), None)
                        
                        if coda_syllable and onset_syllable:
                            # 创建一个虚拟文件名
                            filename = f"{coda_syllable}_{onset_syllable}.wav"
                            # 创建别名
                            alias_item = f"{coda} {onset}"
                            
                            # 处理别名重复
                            count = alias_count[alias_item]
                            alias_unique = alias_item
                            if 0 < count <= max_alternatives:
                                alias_unique = f"{alias_item}#{count}"
                            elif count > max_alternatives:
                                continue
                            
                            alias_count[alias_item] += 1
                            
                            # 添加OTO条目
                            entries.append({
                                'filename': filename,
                                'alias': alias_unique,
                                'offset': 200,  # 默认偏移
                                'consonant': 100,  # 默认值
                                'cutoff': -100,  # 默认值
                                'preutterance': 50,  # 默认值
                                'overlap': 25  # 默认值
                            })
                            
                            print(f"添加未覆盖的coda-onset组合: {alias_item}")
        
        elif mode == "cvc":
            # 检查所有onset-consonant组合
            for onset in all_onsets:
                for consonant in all_consonants:
                    if (onset, consonant) not in processed_combinations:
                        # 找到具有这个onset的音节和具有这个consonant的音节
                        onset_syllable = next((syl for syl, parts in rules.items() if parts['onset'] == onset), None)
                        consonant_syllable = next((syl for syl, parts in rules.items() if parts['consonant'] == consonant), None)
                        
                        if onset_syllable and consonant_syllable:
                            # 创建一个虚拟文件名
                            filename = f"{onset_syllable}_{consonant_syllable}.wav"
                            # 创建别名
                            alias_item = f"{onset} {consonant}"
                            
                            # 处理别名重复
                            count = alias_count[alias_item]
                            alias_unique = alias_item
                            if 0 < count <= max_alternatives:
                                alias_unique = f"{alias_item}#{count}"
                            elif count > max_alternatives:
                                continue
                            
                            alias_count[alias_item] += 1
                            
                            # 添加OTO条目
                            entries.append({
                                'filename': filename,
                                'alias': alias_unique,
                                'offset': 200,  # 默认偏移
                                'consonant': 100,  # 默认值
                                'cutoff': -100,  # 默认值
                                'preutterance': 50,  # 默认值
                                'overlap': 25  # 默认值
                            })
                            
                            print(f"添加未覆盖的onset-consonant组合: {alias_item}")
        
        # 检查所有coda到R的连接
        for coda in all_codas:
            if (coda, "R") not in processed_combinations:
                # 找到具有这个coda的音节
                coda_syllable = next((syl for syl, parts in rules.items() if parts['coda'] == coda), None)
                
                if coda_syllable:
                    # 创建一个虚拟文件名
                    filename = f"{coda_syllable}_R.wav"
                    # 创建别名
                    alias_item = f"{coda} R"
                    
                    # 处理别名重复
                    count = alias_count[alias_item]
                    alias_unique = alias_item
                    if 0 < count <= max_alternatives:
                        alias_unique = f"{alias_item}#{count}"
                    elif count > max_alternatives:
                        continue
                    
                    alias_count[alias_item] += 1
                    
                    # 添加OTO条目
                    entries.append({
                        'filename': filename,
                        'alias': alias_unique,
                        'offset': 200,  # 默认偏移
                        'consonant': 100,  # 默认值
                        'cutoff': -100,  # 默认值
                        'preutterance': 50,  # 默认值
                        'overlap': 25  # 默认值
                    })
                    
                    print(f"添加未覆盖的coda-R组合: {alias_item}")
        
        # 检查所有结尾coda
        for coda in all_codas:
            if ("end", coda) not in processed_combinations:
                # 找到具有这个coda的音节
                coda_syllable = next((syl for syl, parts in rules.items() if parts['coda'] == coda), None)
                
                if coda_syllable:
                    # 创建一个虚拟文件名
                    filename = f"{coda_syllable}.wav"
                    # 创建别名
                    alias_item = coda
                    
                    # 处理别名重复
                    count = alias_count[alias_item]
                    alias_unique = alias_item
                    if 0 < count <= max_alternatives:
                        alias_unique = f"{alias_item}#{count}"
                    elif count > max_alternatives:
                        continue
                    
                    alias_count[alias_item] += 1
                    
                    # 添加OTO条目
                    entries.append({
                        'filename': filename,
                        'alias': alias_unique,
                        'offset': 200,  # 默认偏移
                        'consonant': 100,  # 默认值
                        'cutoff': -100,  # 默认值
                        'preutterance': 50,  # 默认值
                        'overlap': 25  # 默认值
                    })
                    
                    print(f"添加未覆盖的结尾coda: {alias_item}")

# 文件操作函数
def load_language_file(language_file="languages.ini"):
    global text, current_language
    config = configparser.ConfigParser(interpolation=None)
    available_languages = []
    
    try:
        # 获取语言文件绝对路径
        if not os.path.exists(language_file):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            language_file = os.path.join(base_dir, "languages.ini")
            if not os.path.exists(language_file):
                raise FileNotFoundError(f"Language file not found: {language_file}")

        # 读取并验证语言文件
        with open(language_file, 'r', encoding='utf-8') as f:
            config.read_file(f)
        
        available_languages = config.sections()
        
        if not available_languages:
            raise ValueError("No languages defined in language file")
            
        # 验证当前语言是否存在
        if current_language not in config:
            print(f"Warning: Language '{current_language}' not found, falling back to first available")
            current_language = available_languages[0]
        
        # 检查重复键
        text = {}
        for key, value in config[current_language].items():
            if key in text:
                print(f"Warning: Duplicate key '{key}' in language file")
            text[key] = value
        
        return available_languages
        
    except Exception as e:
        print(f"Error loading language file: {e}")
        # 创建基本英语回退
        text = {
            "title": "Recording List Generator",
            "error": "Error",
            "rules_file_label": "Rule File:",
            "settings_label": "Settings",
            # 添加其他必要键...
        }
        return ["en"]

def browse_file():
    filename = filedialog.askopenfilename(
        initialdir=os.getcwd(), 
        title=get_text("browse_button"),
        filetypes=(("INI files", "*.ini"), ("all files", "*.*"))
    )
    if filename:
        rules_file_path.set(filename)

def get_unique_filename(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base}_{counter}{ext}"
        counter += 1
    return filename

def export_reclist():
    try:
        filename = get_unique_filename("reclist.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(recording_list))
        messagebox.showinfo(get_text("success"), get_text("recording_list_saved").format(filename))
    except Exception as e:
        messagebox.showerror(get_text("error"), get_text("failed_to_export_recording_list").format(e))

def export_oto():
    try:
        filename = get_unique_filename("oto.ini")
        with open(filename, "w", encoding="utf-8") as f:
            for entry in oto_entries:
                f.write(f"{entry['filename']}={entry['alias']},{entry['offset']},"
                       f"{entry['consonant']},{entry['cutoff']},"
                       f"{entry['preutterance']},{entry['overlap']}\n")
        messagebox.showinfo(get_text("success"), get_text("oto_configuration_saved").format(filename))
    except Exception as e:
        messagebox.showerror(get_text("error"), get_text("failed_to_export_oto_configuration").format(e))

# 主功能函数
def generate_recording_list_and_oto():
    global recording_list, oto_entries, syllable_counts, \
           max_syllables_entry, bpm_entry, leading_silence_entry, max_alternatives_entry, \
           result_text, oto_text, export_reclist_button, export_oto_button, \
           reclist_line_count_label, oto_line_count_label, generation_mode_map

    # === 强制清空所有旧数据 ===
    recording_list = []
    oto_entries = []
    syllable_counts = {}

    # 清空文本框与状态标签
    result_text.config(state=tk.NORMAL)
    oto_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    oto_text.delete("1.0", tk.END)
    reclist_line_count_label.config(text=get_text("line_count").format(0))
    oto_line_count_label.config(text=get_text("line_count").format(0))

    # === 构建设置参数 ===
    try:
        # 获取生成模式的显示值
        display_generation_mode = generation_mode_var.get()
        # 将显示值转换为实际值
        actual_generation_mode = generation_mode_map.get(display_generation_mode, "none")
        print(f"选择的生成模式: {display_generation_mode} -> {actual_generation_mode}")
        
        settings = {
            'max_syllables_per_sentence': int(max_syllables_entry.get()),
            'mode': mode_var.get(),
            'generation_mode': actual_generation_mode
        }
        oto_params = {
            'bpm': int(bpm_entry.get()),
            'leading_silence': int(leading_silence_entry.get()),
            'max_alternatives': int(max_alternatives_entry.get())
        }

        # === 解析规则文件并生成录音列表与 OTO 条目 ===
        rules = RuleParser.parse(rules_file_path.get())
        recording_list, syllables, syllable_counts = RuleParser.generate_recording_list(rules, settings)
        oto_entries = OTOGenerator.generate(rules, recording_list, oto_params,
                                           oto_params['max_alternatives'],
                                           settings['mode'],
                                           settings['generation_mode'])

        # === 更新 UI 显示 ===
        result_text.insert(tk.END, "\n".join(recording_list))
        oto_output = "\n".join(f"{e['filename']}={e['alias']},{e['offset']},"
                               f"{e['consonant']},{e['cutoff']},"
                               f"{e['preutterance']},{e['overlap']}"
                               for e in oto_entries)
        oto_text.insert(tk.END, oto_output)

        reclist_lines = result_text.index('end-1c').split('.')[0]
        oto_lines = oto_text.index('end-1c').split('.')[0]
        reclist_line_count_label.config(text=get_text("line_count").format(reclist_lines))
        oto_line_count_label.config(text=get_text("line_count").format(oto_lines))

        export_reclist_button.config(state="normal")
        export_oto_button.config(state="normal")

    except Exception as e:
        error_msg = f"{get_text('error')}: {str(e)}"
        result_text.insert(tk.END, error_msg)
        oto_text.insert(tk.END, error_msg)
        reclist_line_count_label.config(text=get_text("line_count").format(0))
        oto_line_count_label.config(text=get_text("line_count").format(0))
        export_reclist_button.config(state="disabled")
        export_oto_button.config(state="disabled")

    finally:
        result_text.config(state=tk.DISABLED)
        oto_text.config(state=tk.DISABLED)


def switch_language(language):
    global current_language
    current_language = language
    load_language_file()
    update_ui_text()

def update_ui_text():
    # 确保所有需要更新的 UI 控件和相关 Text 控件都声明为 global
    global root, settings_frame, notebook, rules_file_label, max_syllables_label, \
           mode_label, generation_mode_label, bpm_label, leading_silence_label, \
           max_alternatives_label, generate_button, export_reclist_button, \
           export_oto_button, browse_button, language_label, \
           mode_combobox, generation_mode_combobox, \
           reclist_line_count_label, oto_line_count_label, result_text, oto_text, \
           generation_mode_map

    # 更新窗口标题和框架标签
    if root:
        root.title(get_text("title"))
    if settings_frame:
        settings_frame.config(text=get_text("settings_label"))
    if notebook:
        # 确保 Notebook 至少有两个标签页
        try:
            notebook.tab(0, text=get_text("recording_list"))
            notebook.tab(1, text=get_text("oto_configuration"))
        except tk.TclError:
             # 如果标签页不存在（例如初始化早期），则忽略
             pass

    # 更新其他控件文本
    widgets_to_update = [
        (rules_file_label, "rules_file_label"),
        (max_syllables_label, "max_syllables_label"),
        (mode_label, "mode_label"),
        (generation_mode_label, "generation_mode_label"),
        (bpm_label, "bpm_label"),
        (leading_silence_label, "leading_silence_label"),
        (max_alternatives_label, "max_alternatives_label"),
        (generate_button, "generate_button"),
        (export_reclist_button, "save_recording_list"),
        (export_oto_button, "save_oto_configuration"),
        (browse_button, "browse_button"),
        (language_label, "language_label")
    ]
    for widget, key in widgets_to_update:
        if widget: # 检查控件是否存在
            try:
                widget.config(text=get_text(key))
            except tk.TclError:
                 # 控件可能尚未完全初始化，忽略错误
                 pass

    # 更新模式选择下拉框选项
    if mode_combobox:
        try:
            mode_combobox['values'] = [
                get_text("mode_vccv_cvvc"),
                get_text("mode_vcv"),
                get_text("mode_cvc")
            ]
        except tk.TclError:
            pass # 忽略错误

    # 更新生成模式下拉框选项
    if generation_mode_combobox:
        try:
            # 更新下拉框选项
            mode_values = [
                get_text("generation_mode_none"),
                get_text("generation_mode_repeat"),
                get_text("generation_mode_interval"),
                get_text("generation_mode_sequence")
            ]
            generation_mode_combobox['values'] = mode_values
            
            # 更新全局映射表
            global generation_mode_map
            generation_mode_map = {
                get_text("generation_mode_none"): "none",
                get_text("generation_mode_repeat"): "repeat",
                get_text("generation_mode_interval"): "interval",
                get_text("generation_mode_sequence"): "sequence"
            }
            print(f"已更新生成模式映射: {generation_mode_map}")
            
            # 如果当前选择的值在新的映射中不存在，则重置为默认值
            current_display = generation_mode_var.get()
            if current_display not in generation_mode_map:
                generation_mode_var.set(get_text("generation_mode_none"))
        except tk.TclError:
            pass # 忽略错误

    # 更新行数标签文本
    line_count_format_string = get_text("line_count") # 获取 "行数: {}" 或 "Lines: {}"

    # 更新录音列表行数标签
    if reclist_line_count_label:
        current_reclist_lines = "0" # 默认行数为 0
        if result_text: # 检查 Text 控件是否存在
            try:
                content = result_text.get("1.0", tk.END).strip()
                if content: # 仅当有内容时计算行数
                    current_reclist_lines = result_text.index('end-1c').split('.')[0]
            except tk.TclError:
                # 控件可能尚未完全初始化或访问错误，保持行数为 0
                pass
        try:
            reclist_line_count_label.config(text=line_count_format_string.format(current_reclist_lines))
        except tk.TclError:
            pass # 忽略错误

    # 更新 OTO 配置行数标签
    if oto_line_count_label:
        current_oto_lines = "0" # 默认行数为 0
        if oto_text: # 检查 Text 控件是否存在
            try:
                content = oto_text.get("1.0", tk.END).strip()
                if content: # 仅当有内容时计算行数
                    current_oto_lines = oto_text.index('end-1c').split('.')[0]
            except tk.TclError:
                # 控件可能尚未完全初始化或访问错误，保持行数为 0
                pass
        try:
            oto_line_count_label.config(text=line_count_format_string.format(current_oto_lines))
        except tk.TclError:
            pass # 忽略错误


def load_config(config_file):
    global max_syllables_entry, mode_var, bpm_entry, leading_silence_entry, max_alternatives_entry # <-- 添加 global 声明
    config = configparser.ConfigParser()
    try:
        config.read(config_file, encoding="utf-8")
        max_syllables_entry.delete(0, tk.END)
        max_syllables_entry.insert(0, config.get("GENERAL", "max_syllables_per_sentence", fallback="8"))
        mode_var.set(config.get("GENERAL", "mode", fallback="vccv-cvvc"))
        bpm_entry.delete(0, tk.END)
        bpm_entry.insert(0, config.get("OTO", "bpm", fallback="120"))
        leading_silence_entry.delete(0, tk.END)
        leading_silence_entry.insert(0, config.get("OTO", "leading_silence", fallback="100"))
        max_alternatives_entry.delete(0, tk.END)
        max_alternatives_entry.insert(0, config.get("OTO", "max_alternatives", fallback="1"))
    except Exception as e:
        print(f"Error loading config: {e}")
        messagebox.showerror(get_text("error"), get_text("failed_to_load_configuration_file").format(e))

def initialize_ui():
    global root, rules_file_label, max_syllables_label, mode_label, generation_mode_label, bpm_label, \
           leading_silence_label, max_alternatives_label, generate_button, \
           export_reclist_button, export_oto_button, browse_button, \
           settings_frame, notebook, reclist_frame, oto_frame, \
           result_text, oto_text, button_frame, language_label, language_combobox, \
           max_syllables_entry, bpm_entry, leading_silence_entry, max_alternatives_entry, \
           mode_combobox, generation_mode_combobox, \
           reclist_line_count_label, oto_line_count_label # <-- 添加行数标签到 global

    root.deiconify()
    root.title(get_text("title"))
    root.geometry("800x700")

    # 主框架
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    # 配置主框架的列权重，让设置区域和预览区域合理分配空间
    main_frame.columnconfigure(0, weight=1) # 左侧列（设置）
    main_frame.columnconfigure(1, weight=0) # 中间列（按钮）
    main_frame.columnconfigure(2, weight=0) # 右侧列（按钮）
    main_frame.rowconfigure(3, weight=1) # 让 Notebook 区域垂直扩展

    # 规则文件选择
    rules_file_label = ttk.Label(main_frame)
    rules_file_entry = ttk.Entry(main_frame, textvariable=rules_file_path, state="readonly")
    browse_button = ttk.Button(main_frame, command=browse_file)

    # 参数设置区域
    settings_frame = ttk.LabelFrame(main_frame)
    # 配置 settings_frame 内部的列权重
    settings_frame.columnconfigure(1, weight=1)
    settings_frame.columnconfigure(3, weight=1)

    max_syllables_label = ttk.Label(settings_frame)
    max_syllables_entry = ttk.Entry(settings_frame, width=10) # 限制宽度
    mode_label = ttk.Label(settings_frame)
    mode_combobox = ttk.Combobox(settings_frame, textvariable=mode_var,
                                 values=["vccv-cvvc", "vcv", "cvc"], state="readonly", width=15) # 限制宽度

    generation_mode_label = ttk.Label(settings_frame)
    generation_mode_combobox = ttk.Combobox(settings_frame, textvariable=generation_mode_var,
                                            values=["none", "repeat", "interval", "sequence"],
                                            state="readonly", width=15) # 限制宽度

    bpm_label = ttk.Label(settings_frame)
    bpm_entry = ttk.Entry(settings_frame, width=10) # 限制宽度
    leading_silence_label = ttk.Label(settings_frame)
    leading_silence_entry = ttk.Entry(settings_frame, width=10) # 限制宽度
    max_alternatives_label = ttk.Label(settings_frame)
    max_alternatives_entry = ttk.Entry(settings_frame, width=10) # 限制宽度

    generate_button = ttk.Button(main_frame, command=generate_recording_list_and_oto)

    # 结果展示区域 (Notebook)
    notebook = ttk.Notebook(main_frame)

    # --- 录音列表 Tab ---
    reclist_frame = ttk.Frame(notebook)
    # 配置 reclist_frame 内部的行列权重
    reclist_frame.grid_rowconfigure(0, weight=1)    # 让 Text 区域垂直填充
    reclist_frame.grid_columnconfigure(0, weight=1) # 让 Text 区域水平填充

    # 创建滚动条
    reclist_scrollbar = ttk.Scrollbar(reclist_frame, orient=tk.VERTICAL)
    # 创建 Text 控件，并关联滚动条
    result_text = tk.Text(reclist_frame, wrap="none", yscrollcommand=reclist_scrollbar.set, state=tk.DISABLED, height=10) # 初始设为禁用, wrap="none"
    # 配置滚动条命令
    reclist_scrollbar.config(command=result_text.yview)

    # 布局 Text 和 Scrollbar
    result_text.grid(row=0, column=0, sticky="nsew")
    reclist_scrollbar.grid(row=0, column=1, sticky="ns")

    # 创建行数标签
    reclist_line_count_label = ttk.Label(reclist_frame, text=get_text("line_count").format(0))
    reclist_line_count_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2) # 放在下方

    # --- OTO 配置 Tab ---
    oto_frame = ttk.Frame(notebook)
    # 配置 oto_frame 内部的行列权重
    oto_frame.grid_rowconfigure(0, weight=1)    # 让 Text 区域垂直填充
    oto_frame.grid_columnconfigure(0, weight=1) # 让 Text 区域水平填充

    # 创建滚动条
    oto_scrollbar = ttk.Scrollbar(oto_frame, orient=tk.VERTICAL)
    # 创建 Text 控件，并关联滚动条
    oto_text = tk.Text(oto_frame, wrap="none", yscrollcommand=oto_scrollbar.set, state=tk.DISABLED, height=10) # 初始设为禁用, wrap="none"
    # 配置滚动条命令
    oto_scrollbar.config(command=oto_text.yview)

    # 布局 Text 和 Scrollbar
    oto_text.grid(row=0, column=0, sticky="nsew")
    oto_scrollbar.grid(row=0, column=1, sticky="ns")

    # 创建行数标签
    oto_line_count_label = ttk.Label(oto_frame, text=get_text("line_count").format(0))
    oto_line_count_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2) # 放在下方

    # 将 Frame 添加到 Notebook
    notebook.add(reclist_frame, text="") # 文本将在 update_ui_text 中设置
    notebook.add(oto_frame, text="")   # 文本将在 update_ui_text 中设置

    # 底部按钮区域
    button_frame = ttk.Frame(main_frame)
    export_reclist_button = ttk.Button(button_frame, command=export_reclist, state="disabled")
    export_oto_button = ttk.Button(button_frame, command=export_oto, state="disabled")

    language_label = ttk.Label(button_frame)
    # 假设 available_languages 在调用此函数前已加载
    language_combobox = ttk.Combobox(button_frame, textvariable=language_var,
                                     values=available_languages, state="readonly", width=5)

    # --- 布局UI控件 ---
    # 规则文件行
    rules_file_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
    rules_file_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=(0, 5))
    browse_button.grid(row=0, column=2, padx=5, pady=(0, 5))

    # 设置框架
    settings_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=5, padx=0)
    # 设置框架内部控件布局
    pad_x = (10, 5)
    pad_y = 2
    max_syllables_label.grid(row=0, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
    max_syllables_entry.grid(row=0, column=1, sticky=tk.EW, padx=pad_x, pady=pad_y)
    mode_label.grid(row=0, column=2, sticky=tk.W, padx=pad_x, pady=pad_y)
    mode_combobox.grid(row=0, column=3, sticky=tk.EW, padx=pad_x, pady=pad_y)

    generation_mode_label.grid(row=1, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
    generation_mode_combobox.grid(row=1, column=1, sticky=tk.EW, padx=pad_x, pady=pad_y)
    bpm_label.grid(row=1, column=2, sticky=tk.W, padx=pad_x, pady=pad_y)
    bpm_entry.grid(row=1, column=3, sticky=tk.EW, padx=pad_x, pady=pad_y)

    leading_silence_label.grid(row=2, column=0, sticky=tk.W, padx=pad_x, pady=pad_y)
    leading_silence_entry.grid(row=2, column=1, sticky=tk.EW, padx=pad_x, pady=pad_y)
    max_alternatives_label.grid(row=2, column=2, sticky=tk.W, padx=pad_x, pady=pad_y)
    max_alternatives_entry.grid(row=2, column=3, sticky=tk.EW, padx=pad_x, pady=pad_y)

    # 生成按钮
    generate_button.grid(row=2, column=0, columnspan=3, pady=10)

    # Notebook (预览区域)
    notebook.grid(row=3, column=0, columnspan=3, sticky=tk.NSEW, pady=5)

    # 底部按钮框架
    button_frame.grid(row=4, column=0, columnspan=3, pady=(10, 0), sticky="ew") # 使用 sticky="ew" 使其水平填充
    # 底部按钮框架内部控件布局 (使用 pack)
    export_reclist_button.pack(side=tk.LEFT, padx=5)
    export_oto_button.pack(side=tk.LEFT, padx=5)
    # 将语言选择放在右侧
    language_combobox.pack(side=tk.RIGHT, padx=5)
    language_label.pack(side=tk.RIGHT, padx=5)
    language_combobox.bind("<<ComboboxSelected>>", lambda e: switch_language(language_var.get()))

    # 初始化值
    max_syllables_entry.insert(0, "6")
    bpm_entry.insert(0, "120")
    leading_silence_entry.insert(0, "100")
    max_alternatives_entry.insert(0, "3")
    language_var.set(current_language)

    update_ui_text() # 更新所有文本
    load_config(config_file) # 加载配置覆盖默认值

    # 确保主框架的行和列配置正确以允许扩展
    main_frame.grid_rowconfigure(3, weight=1) # Notebook 行
    main_frame.grid_columnconfigure(1, weight=1) # Entry 列

    root.deiconify() # 显示窗口
    root.mainloop()

# --- 程序入口 ---
if __name__ == "__main__":
    available_languages = load_language_file()
    initialize_ui() # 调用初始化函数