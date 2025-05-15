# spg_parser.py (AI-centric for next logic and redundant details - CORRECTED INDENTATION)
import json
import logging
import re
from functools import cmp_to_key
from typing import Any, Dict, List, Optional, Tuple

# Import the REAL AI client functions
import ai_client

# --- Constants ---
METADATA_SEARCH_LIMIT = 6000
EXPECTED_QUESTION_KEYS = {"id", "question", "type", "options", "hovertexts", "probes", "next", "validations", "programming_details"}

# Configure logging
logger = logging.getLogger(__name__)

# --- Text Processing Functions ---
def split_text_for_metadata(full_text: str) -> Tuple[Optional[str], Optional[str]]:
    if not full_text or len(full_text) < 50:
        logger.debug("Full text too short or empty for metadata split.")
        return None, full_text

    search_limit = min(len(full_text), METADATA_SEARCH_LIMIT)
    last_saved_label_pos = full_text.find('Last Saved:', 0, search_limit)
    qnum_pos = full_text.find('Qnum', 0, search_limit)
    last_saved_end_pos = -1
    if last_saved_label_pos != -1:
        eol_after_last_saved = full_text.find('\n', last_saved_label_pos, search_limit)
        if eol_after_last_saved != -1:
            last_saved_end_pos = eol_after_last_saved + 1
        else:
            last_saved_end_pos = min(search_limit, last_saved_label_pos + 50)

    default_split_point = min(len(full_text), 3000)
    split_point = default_split_point
    if qnum_pos != -1 and qnum_pos < default_split_point + 1000:
        split_point = qnum_pos
        if last_saved_end_pos != -1 and last_saved_end_pos > split_point:
            split_point = last_saved_end_pos
    elif last_saved_end_pos != -1:
        split_point = last_saved_end_pos
    else:
        first_para_break_after_default = full_text.find('\n\n', default_split_point, search_limit)
        if first_para_break_after_default != -1:
            split_point = first_para_break_after_default + 2

    split_point = max(0, min(split_point, len(full_text)))
    metadata_chunk = full_text[:split_point].strip()
    remaining_text = full_text[split_point:].strip()

    if not metadata_chunk:
        logger.warning("Metadata chunk is empty after split attempt.")
        return None, full_text
    if not remaining_text:
        logger.warning("Remaining text is empty after metadata split.")
        return metadata_chunk, None
    logger.debug(f"Split: Metadata chunk (len {len(metadata_chunk)}), Remaining (len {len(remaining_text)}).")
    return metadata_chunk, remaining_text

def create_text_batches(text: str, target_chunk_size: int = 10000) -> List[str]:
    batches = []
    current_pos = 0
    text_len = len(text)
    if not text:
        return []
    while current_pos < text_len:
        end_pos = min(current_pos + target_chunk_size, text_len)
        if end_pos < text_len:
            para_break = text.rfind('\n\n', max(0, end_pos - 1000), end_pos)
            if para_break != -1 and para_break > current_pos:
                end_pos = para_break + 2
            else:
                line_break = text.rfind('\n', max(0, end_pos - 500), end_pos)
                if line_break != -1 and line_break > current_pos:
                    end_pos = line_break + 1
        chunk = text[current_pos:end_pos].strip()
        if chunk:
            batches.append(chunk)
        if end_pos <= current_pos:
            logger.error(f"Batch splitting error: end_pos {end_pos} <= current_pos {current_pos}. Stopping.")
            break
        current_pos = end_pos
    logger.info(f"Split text (len {text_len}) into {len(batches)} batches.")
    return batches

def _clean_and_parse_json_list(json_text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not json_text:
        logger.warning("Received empty or None JSON text for parsing.")
        return None
    try:
        cleaned_text = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', json_text.strip(), flags=re.MULTILINE | re.DOTALL).strip()
        if cleaned_text.lower().startswith('json'):
            cleaned_text = cleaned_text[4:].lstrip()
        try:
            parsed_json = json.loads(cleaned_text)
            if isinstance(parsed_json, list):
                return parsed_json
            if isinstance(parsed_json, dict):
                logger.warning("Found single JSON object instead of a list. Wrapping it.")
                return [parsed_json]
            else:
                logger.error(f"Parsed JSON is not a list or dict, but type: {type(parsed_json)}")
                return None
        except json.JSONDecodeError:
            start_index = cleaned_text.find('[')
            end_index = cleaned_text.rfind(']')
            if start_index != -1 and end_index != -1 and end_index >= start_index:
                json_portion = cleaned_text[start_index : end_index + 1]
                try:
                    parsed_json_list = json.loads(json_portion)
                    if isinstance(parsed_json_list, list):
                        return parsed_json_list
                    else:
                        logger.error(f"Parsed JSON portion '[]' is not a list, but type: {type(parsed_json_list)}")
                        return None
                except json.JSONDecodeError as e_inner:
                    problem_start = max(0, e_inner.pos - 50)
                    problem_end = min(len(json_portion), e_inner.pos + 50)
                    logger.error(f"JSONDecodeError within '[]': {e_inner}. Problem text near pos {e_inner.pos}: ...{json_portion[problem_start:problem_end]}...")
                    return None
            else:
                logger.error(f"Could not find valid JSON list delimiters '[' and ']' in cleaned text: {cleaned_text[:200]}...")
                return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON parsing: {e}", exc_info=True)
        return None

# --- Prompt Generation Functions ---
def _generate_metadata_prompt(text_chunk: str) -> str:
    prompt = f"""Analyze the following text from the beginning of a Survey Programming Guide (SPG) document. Extract the specified metadata fields. Respond ONLY with a single JSON object. If a field is not found, use a null value.
Required fields: "name", "version", "survey_number", "last_saved".
Example Output: {{ "name": "Survey Title", "version": 1, "survey_number": 123, "last_saved": "YYYY-MM-DD" }}
Input Text:
---
{text_chunk}
---
JSON Output:
"""
    return prompt

def _generate_question_batch_prompt(text_batch: str) -> str:
    prompt = f"""
**Task:**
Analyze the 'Input SPG Text Batch'. Identify each distinct survey question section (starting with a Qnum). Extract information for **actual questions** presented to respondents.

**Input Format Notes:**
- The input may contain tables formatted using Markdown syntax, enclosed by `**DOCX_TABLE_START**` and `**DOCX_TABLE_END**`. Treat the content within these tables as part of the survey structure. Grid questions are often represented this way.

**CRITICAL Instructions for AI - Parsing Logic and Output Schema:**
-   Identify blocks by Qnum.
-   Create ONE JSON object ONLY for each identified ACTUAL question section.
-   Process actual questions strictly in the order they appear and maintain original order in the output list.
-   If no actual question sections are identified, return an empty JSON list `[]`.

**Field-Specific Instructions:**

1.  **`id`**: (string) Qnum (e.g., "540.0010"). Required.
2.  **`question`**: (string) Respondent-facing question text. May include text from tables. Replace newlines with spaces.
3.  **`type`**: (string) Question type: "select_one", "select_multiple", "numeric", "percentage", "currency", "text", "presentation_text", "grid_select_one", "grid_select_multiple", or "unknown".
    *   If the type is "numeric", "percentage", "currency", or "text", DO NOT include basic type instructions like "PROGRAMMING INSTRUCTIONS: Number" or "PROGRAMMING INSTRUCTIONS: Text" in the `programming_details` field.
4.  **`options`**: (array of objects `{{ "v": any, "t": "string" }}`) Options array for select/grid types. For grids, 'v' might be row identifier, 't' the row text. Empty `[]` if none.
5.  **`hovertexts`**: (string or null) Find "HOVERTEXT:". Extract the entire text block following it until the next major section (like "PROGRAMMING INSTRUCTIONS:") as a SINGLE RAW STRING. Replace internal newlines with spaces. If no "HOVERTEXT:" found, use null or an empty string.
6.  **`probes`**: (object `{{}}`) Default to empty dictionary. (Reserved for future use).
7.  **`next`**: (object `{{ "condition": "Action" }}`) **CRITICAL FOR THIS FIELD:**
    *   Analyze text within "PROGRAMMING INSTRUCTIONS" or similar sections for skip logic.
    *   Convert natural language skip logic into a structured `next` object.
    *   **Examples of conversions:**
        *   "If respondent selects 2 “No”, DQ from survey"  => `{{ "2": "@DQ" }}`
        *   "If response is 1, go to Q101.0020" => `{{ "1": "@Q[101.0020]" }}`
        *   "If answer for 540.0110 is 23, then ask 540.0120" (This describes logic for 540.0110 to go to 540.0120) => For Q 540.0110, this would be `{{ "23": "@Q[540.0120]" }}`.
        *   "SHOW Q540.0300 if respondent selects 11 “Other (please specify)” on previous question (Q540.0290)" => For Q540.0290, this would be `{{ "11": "@Q[540.0300]" }}`.
        *   "Default skip to next question" or if no other logic applies => `{{ "@DEFAULT": "@NEXT" }}`
        *   "If blank, go to next question" => `{{ "@BLANK": "@NEXT" }}` (or if sequential is default, this might be empty)
    *   Use "@Q[QNUM_HERE]" for specific question targets (e.g., "@Q[540.0120]").
    *   Use "@DQ" for disqualification. Use "@NEXT" for sequential. Use "@END" for end of survey.
    *   If multiple conditions exist for a question, include all in the object.
    *   If no skip logic is found for a question, use an empty object `{{}}` or `{{ "@DEFAULT": "@NEXT" }}` if sequential flow is implied.
8.  **`validations`**: (array of objects `[ {{ "rule": "string", "msg": "string or null" }} ]`) Extract validation rules and their messages.
    *   Example: "NOTE: If response > 195, please display... “error message”" => `[ {{ "rule": ">195", "msg": "error message" }} ]`
    *   Example: "Response must be between 0-100%" => `[ {{ "rule": "0-100%", "msg": null }} ]`
9.  **`programming_details`**: (array of strings `[ "string" ]`)
    *   Include ONLY programming instructions that are NOT captured by other fields (like `type`, `next`, or `validations`).
    *   **DO NOT include redundant instructions** like "PROGRAMMING INSTRUCTIONS: Select Only One" if `type` is already "select_one".
    *   DO NOT include skip logic instructions here if they have been successfully converted into the `next` field.
    *   Keep instructions about randomization, table arrangement (e.g., "Please arrange QX through QY in a table..."), or complex inter-question dependencies that cannot be easily put into `next` or `validations`.

**Output Format:**
Respond ONLY with a single JSON list `[...]` containing objects for **actual questions**, adhering to the schema and instructions above. Replace internal newlines in all string values with spaces.

Input SPG Text Batch:
---
{text_batch}
---
JSON Output:
"""
    return prompt

# --- Post-Processing and Validation Functions ---
def _clean_newlines_in_data(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _clean_newlines_in_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_clean_newlines_in_data(elem) for elem in data]
    elif isinstance(data, str):
        cleaned_string = re.sub(r'[\n\r\t]+', ' ', data)
        cleaned_string = re.sub(r' +', ' ', cleaned_string).strip()
        return cleaned_string
    else:
        return data

def _validate_and_conform_question_data(parsed_items: List[Dict[str, Any]], batch_identifier: str) -> List[Dict[str, Any]]:
    conformed_items = []
    if not isinstance(parsed_items, list):
        logger.error(f"[{batch_identifier}] Expected list for validation, got {type(parsed_items)}. Skipping.")
        return []

    for i, item in enumerate(parsed_items):
        q_id_for_log = "UnknownID"
        original_item_str_preview = str(item)[:200]
        if not isinstance(item, dict):
            logger.warning(f"[{batch_identifier}] Item {i} is not a dictionary. Skipping: {original_item_str_preview}...")
            continue

        try:
            item = _clean_newlines_in_data(item)
        except Exception as e:
            logger.error(f"[{batch_identifier}] Error cleaning newlines for item {i}: {e}. Proceeding.", exc_info=True)

        q_id = item.get('id')
        if not q_id or not isinstance(q_id, str) or not q_id.strip():
            logger.error(f"[{batch_identifier}] Item {i} missing/invalid 'id'. Skipping. Item: {original_item_str_preview}")
            continue
        q_id_for_log = q_id

        conformed_item = {'id': q_id}
        conformed_item['question'] = str(item.get('question', '')).strip()
        q_type_raw = item.get('type')
        conformed_item['type'] = q_type_raw.strip().lower() if isinstance(q_type_raw, str) and q_type_raw.strip() else "unknown"
        if conformed_item['type'] == "unknown" and q_type_raw is not None:
            logger.warning(f"[{batch_identifier} - Q {q_id_for_log}] Type invalid ('{q_type_raw}'), defaulted 'unknown'.")

        options = item.get('options')
        valid_options = []
        if isinstance(options, list):
            for opt_idx, opt in enumerate(options):
                if isinstance(opt, dict) and 'v' in opt and 't' in opt:
                    valid_options.append({"v": opt['v'], "t": str(opt.get('t', "")).strip()})
                else:
                    logger.warning(f"[{batch_identifier} - Q {q_id_for_log}] Invalid option {opt_idx}: {str(opt)[:100]}. Skipping.")
            conformed_item['options'] = valid_options
        else:
            conformed_item['options'] = []

        hovertexts_raw = item.get('hovertexts')
        if isinstance(hovertexts_raw, str) and hovertexts_raw.strip():
            conformed_item['hovertexts'] = hovertexts_raw.strip()
        else:
            conformed_item['hovertexts'] = None

        probes_raw = item.get('probes')
        conformed_item['probes'] = {str(k).strip(): str(v).strip() for k, v in probes_raw.items() if str(k).strip()} if isinstance(probes_raw, dict) else {}
        
        next_logic_raw = item.get('next')
        if isinstance(next_logic_raw, dict):
            conformed_item['next'] = {str(k).strip(): str(v).strip() for k, v in next_logic_raw.items() if str(k).strip()}
        elif next_logic_raw is not None:
            logger.warning(f"[{batch_identifier} - Q {q_id_for_log}] 'next' field from AI was not a dict (type: {type(next_logic_raw)}). Defaulting to empty dict. Raw value: {str(next_logic_raw)[:100]}")
            conformed_item['next'] = {}
        else:
            conformed_item['next'] = {}

        validations_raw = item.get('validations')
        valid_validations = []
        if isinstance(validations_raw, list):
            for val_idx, val_item in enumerate(validations_raw):
                if isinstance(val_item, dict) and 'rule' in val_item:
                    rule_text = str(val_item.get('rule', "")).strip()
                    msg_val = val_item.get('msg')
                    msg_text = str(msg_val).strip() if msg_val is not None else None
                    if rule_text:
                        valid_validations.append({"rule": rule_text, "msg": msg_text})
                else:
                    logger.warning(f"[{batch_identifier} - Q {q_id_for_log}] Invalid validation {val_idx}: {str(val_item)[:100]}. Skipping.")
        conformed_item['validations'] = valid_validations

        prog_details_raw = item.get('programming_details')
        if isinstance(prog_details_raw, list):
            conformed_item['programming_details'] = [str(pd).strip() for pd in prog_details_raw if str(pd).strip()]
        elif prog_details_raw is not None:
            logger.warning(f"[{batch_identifier} - Q {q_id_for_log}] 'programming_details' not list (got {type(prog_details_raw)}). Defaulting [].")
            conformed_item['programming_details'] = []
        else:
            conformed_item['programming_details'] = []

        if conformed_item['type'] in ["select_one", "select_multiple", "grid_select_one", "grid_select_multiple"] and not conformed_item['options']:
            has_standard_options_note = any("standard options" in str(detail).lower() for detail in conformed_item.get('programming_details', []))
            if not has_standard_options_note:
                logger.warning(f"[{batch_identifier} - Q {q_id_for_log}] Type '{conformed_item['type']}' but 'options' empty & no 'standard options' note in programming_details.")

        for key in EXPECTED_QUESTION_KEYS:
            if key not in conformed_item:
                logger.debug(f"[{batch_identifier} - Q {q_id_for_log}] Key '{key}' missing. Adding default.")
                defaults = {"options": [], "validations": [], "programming_details": [], "hovertexts": None, "probes": {}, "next": {}, "type": "unknown", "question": ""}
                if key in defaults:
                    conformed_item[key] = defaults[key]
        conformed_items.append(conformed_item)
    return conformed_items

def _consolidate_duplicate_qids(parsed_questions: List[Dict[str, Any]], batch_identifier: str) -> List[Dict[str, Any]]:
    questions_by_id = {}
    qids_in_order = []
    for item in parsed_questions:
        q_id = item.get('id')
        if not q_id:
            continue

        is_likely_instruction = (
            (not item.get('question', '').strip() or item.get('question','').lower().strip() == 'none' or 
             item.get('question','').lower().strip().startswith(("please arrange", "show ", "programming note"))) and 
            item.get('type', 'unknown') in ['unknown', 'presentation_text']
        )

        if q_id not in questions_by_id:
            questions_by_id[q_id] = item
            qids_in_order.append(q_id)
        else:
            logger.warning(f"[{batch_identifier}] Duplicate QID '{q_id}'. Consolidating.")
            existing_item = questions_by_id[q_id]
            existing_is_instruction = (
                (not existing_item.get('question', '').strip() or existing_item.get('question','').lower().strip() == 'none' or 
                 existing_item.get('question','').lower().strip().startswith(("please arrange", "show ", "programming note"))) and 
                existing_item.get('type', 'unknown') in ['unknown', 'presentation_text']
            )

            chosen_item = existing_item
            discarded_item = item
            if existing_is_instruction and not is_likely_instruction:
                chosen_item = item
                discarded_item = existing_item
            elif existing_is_instruction == is_likely_instruction:
                if existing_item.get('type') == 'unknown' and item.get('type') != 'unknown':
                    chosen_item = item
                    discarded_item = existing_item
                elif len(item.get('question', '')) > len(existing_item.get('question', '')):
                    chosen_item = item
                    discarded_item = existing_item
            
            if chosen_item != existing_item:
                questions_by_id[q_id] = chosen_item
                logger.info(f"[{batch_identifier}] Swapped preferred item for QID {q_id}.")

            # Merge details
            chosen_details_set = set(chosen_item.get('programming_details', []))
            discarded_details_set = set(discarded_item.get('programming_details', []))
            chosen_item['programming_details'] = sorted(list(chosen_details_set.union(discarded_details_set)))
            
            merged_validations = {}
            for v in chosen_item.get('validations', []):
                rule = v.get('rule')
                if rule:
                    merged_validations[rule] = v.get('msg')
            for val_disc in discarded_item.get('validations', []):
                rule_disc = val_disc.get('rule')
                if rule_disc and rule_disc not in merged_validations:
                    merged_validations[rule_disc] = val_disc.get('msg')
            chosen_item['validations'] = [{"rule": r, "msg": m} for r, m in merged_validations.items()]
            
            merged_next = chosen_item.get('next', {}).copy()
            merged_next.update(discarded_item.get('next', {}))
            chosen_item['next'] = merged_next
            
            chosen_ht_val = chosen_item.get('hovertexts')
            discarded_ht_val = discarded_item.get('hovertexts')
            if not chosen_ht_val and discarded_ht_val:
                chosen_item['hovertexts'] = discarded_ht_val
            
            merged_probes = chosen_item.get('probes', {}).copy()
            merged_probes.update(discarded_item.get('probes', {}))
            chosen_item['probes'] = merged_probes
            
            if chosen_item == item: # If 'item' was chosen, its options are preferred
                chosen_item['options'] = item.get('options', [])

    return [questions_by_id[qid_val] for qid_val in qids_in_order]

def parse_spg_batches(text_batches: List[str], run_id: str = "N/A") -> List[Dict[str, Any]]:
    all_parsed_questions = []
    for i, text_chunk in enumerate(text_batches):
        batch_id_str = f"[Run:{run_id} Batch {i+1}/{len(text_batches)}]"
        logger.info(f"Processing {batch_id_str} (Len: {len(text_chunk)})")
        print(f"-- Processing {batch_id_str} --")
        prompt = _generate_question_batch_prompt(text_chunk)
        raw_response = ai_client.generate_with_gemini(prompt)
        if not raw_response:
            logger.error(f"{batch_id_str} AI call failed or returned no response. Skipping batch.")
            print(f"Error: {batch_id_str} AI failed.")
            continue
        
        logger.debug(f"[{batch_id_str}] Raw AI response (first 200 chars): {raw_response[:200]}")
        parsed_batch_results = _clean_and_parse_json_list(raw_response)
        if parsed_batch_results is None:
            logger.error(f"{batch_id_str} Failed parse AI JSON response. Skipping batch.")
            print(f"Error: {batch_id_str} Failed JSON parse.")
            logger.debug(f"{batch_id_str} Raw response causing parse error:\n{raw_response}")
            continue
        
        validated_questions = _validate_and_conform_question_data(parsed_batch_results, batch_id_str)
        consolidated_batch_questions = _consolidate_duplicate_qids(validated_questions, batch_id_str)
        all_parsed_questions.extend(consolidated_batch_questions)
        logger.info(f"{batch_id_str} complete. Found {len(consolidated_batch_questions)} questions in batch.")
        print(f"{batch_id_str} complete.")
    return all_parsed_questions

# --- Sorting Helpers ---
def qid_sort_key(qid_str: str):
    if not isinstance(qid_str, str): return (float('inf'), str(qid_str))
    match = re.match(r'(\d+)\.?(\d*)?(.*)', qid_str.strip())
    if match:
        major, minor, suffix = match.groups()
        try:
            return (int(major), int(minor) if minor else 0, suffix or '')
        except ValueError:
            return (float('inf'), qid_str)
    try:
        return (int(qid_str), 0, '')
    except ValueError:
        logger.warning(f"Unexpected QID format for sorting: '{qid_str}'.")
        return (float('inf'), qid_str)

def compare_qids(item1_dict, item2_dict):
    key1 = qid_sort_key(item1_dict.get('id', ''))
    key2 = qid_sort_key(item2_dict.get('id', ''))
    if key1 < key2: return -1
    if key1 > key2: return 1
    return 0

# --- Grouping Instruction Propagation ---
def _parse_qid_range(instruction_text: str) -> Tuple[Optional[str], Optional[str]]:
    patterns = [r'(?:arrange|show|ask)\s+(?:questions?\s+)?\b((?:\d+\.)?\d+)\s+(?:through|to|and)\s+\b((?:\d+\.)?\d+)\b']
    for p in patterns:
        match = re.search(p, instruction_text, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
    return None, None

def _propagate_grouping_instructions(sorted_questions: List[Dict[str, Any]], run_id: str = "N/A") -> List[Dict[str, Any]]:
    if not sorted_questions:
        return []
    logger.info(f"[{run_id}] Starting instruction propagation...")
    propagation_map = {}
    for i, current_q in enumerate(sorted_questions):
        for detail_str in current_q.get('programming_details', []):
            start_qid_str, end_qid_str = _parse_qid_range(detail_str)
            if start_qid_str and end_qid_str:
                logger.info(f"[{run_id}] Found grouping instruction '{detail_str[:80]}...' in QID {current_q['id']}")
                start_key = qid_sort_key(start_qid_str)
                end_key = qid_sort_key(end_qid_str)
                start_idx = -1
                for idx_find, q_find in enumerate(sorted_questions):
                    if qid_sort_key(q_find['id']) >= start_key:
                        start_idx = idx_find
                        break
                if start_idx != -1:
                    propagated_count = 0
                    for j in range(start_idx, len(sorted_questions)):
                        target_q_loop = sorted_questions[j]
                        target_key = qid_sort_key(target_q_loop['id'])
                        if target_key > end_key:
                            break
                        if start_key <= target_key <= end_key:
                            if j not in propagation_map:
                                propagation_map[j] = []
                            if detail_str not in propagation_map[j]:
                                propagation_map[j].append(detail_str)
                                propagated_count += 1
                    logger.debug(f"[{run_id}] Marked instruction for {propagated_count} questions.")
                else:
                    logger.warning(f"[{run_id}] Could not find start QID {start_qid_str} for propagation.")
    
    for index, instructions in propagation_map.items():
        if index < len(sorted_questions):
            target_q_apply = sorted_questions[index]
            if 'programming_details' not in target_q_apply or not isinstance(target_q_apply['programming_details'], list):
                target_q_apply['programming_details'] = []
            for instr in instructions:
                if instr not in target_q_apply['programming_details']:
                    target_q_apply['programming_details'].append(instr)
    logger.info(f"[{run_id}] Finished instruction propagation.")
    return sorted_questions

# --- Hovertext Parsing Helper ---
def _parse_raw_hovertexts(questions_list: List[Dict[str, Any]], run_id: str = "N/A") -> List[Dict[str, Any]]:
    logger.info(f"[{run_id}] Starting Python-based hovertext parsing...")
    processed_count = 0
    parsed_count = 0
    term_def_pattern = re.compile(r"([\w\s\(\)\*\.\/&'-]+?)\s*[:*-]\s+(.*?)(?=\n\s*\n|\n?\**[\w\s\(\)\*\.\/&'-]+\**\s*[:*-]\s+|\Z)", re.DOTALL | re.IGNORECASE)
    for question in questions_list:
        raw_hovertext_str = question.get('hovertexts')
        parsed_dict = {}
        if isinstance(raw_hovertext_str, str) and raw_hovertext_str.strip():
            processed_count += 1
            cleaned_block = _clean_newlines_in_data(raw_hovertext_str)
            matches = term_def_pattern.findall(cleaned_block)
            if matches:
                for term, definition in matches:
                    term_cleaned = re.sub(r'^\W+|\W+$', '', term).strip().replace('*', '')
                    definition_cleaned = definition.strip()
                    if term_cleaned and definition_cleaned and len(term_cleaned) > 1 and len(definition_cleaned) > 3:
                        if term_cleaned not in parsed_dict:
                            parsed_dict[term_cleaned] = definition_cleaned
                parsed_count += 1 if parsed_dict else 0
                if not parsed_dict and matches:
                    parsed_dict["raw_hovertext_block"] = cleaned_block
            else: # No regex matches, store raw
                if cleaned_block: # Only store if there's something to store
                     parsed_dict["raw_hovertext_block"] = cleaned_block
            question['hovertexts'] = parsed_dict
        else:
            question['hovertexts'] = {} # Ensure field is dict if originally null/empty/invalid
        
        if 'probes' not in question or not isinstance(question.get('probes'), dict):
            question['probes'] = {}
            
    logger.info(f"[{run_id}] Finished Python hovertext parsing. Processed {processed_count} strings, parsed {parsed_count} into dicts.")
    return questions_list

# --- Main Parsing Function ---
def parse_spg(full_text: str, batch_char_size: int, run_id: str = "N/A") -> Optional[Dict[str, Any]]:
    if not ai_client.is_client_configured():
        logger.critical(f"[{run_id}] SPG Parsing Aborted: AI client is not configured.")
        return None
    logger.info(f"[{run_id}] Starting REAL SPG parsing process...")

    metadata_text_chunk, remaining_text = split_text_for_metadata(full_text)
    metadata = {}
    if metadata_text_chunk:
        logger.info(f"[{run_id}] Attempting REAL metadata extraction...")
        metadata_prompt = _generate_metadata_prompt(metadata_text_chunk)
        metadata_response = ai_client.generate_with_gemini(metadata_prompt)
        if metadata_response:
            try:
                cleaned_meta_text = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', metadata_response.strip(), flags=re.MULTILINE | re.DOTALL).strip()
                if cleaned_meta_text.lower().startswith('json'):
                    cleaned_meta_text = cleaned_meta_text[4:].lstrip()
                parsed_meta = json.loads(cleaned_meta_text)
                if isinstance(parsed_meta, dict):
                    metadata = parsed_meta
                    logger.info(f"[{run_id}] Metadata extracted: {metadata}")
                    print(f"Info: [{run_id}] Metadata extracted successfully.")
                else:
                    meta_list = _clean_and_parse_json_list(f"[{cleaned_meta_text}]" if not cleaned_meta_text.startswith('[') else cleaned_meta_text)
                    if meta_list and isinstance(meta_list[0], dict):
                        metadata = meta_list[0]
                        logger.info(f"[{run_id}] Metadata extracted (from list): {metadata}")
                        print(f"Info: [{run_id}] Metadata extracted successfully.")
                    else:
                        logger.warning(f"[{run_id}] Failed to parse metadata JSON (not dict/list): {metadata_response[:200]}")
                        print(f"Warning: [{run_id}] Could not parse metadata.")
                        metadata = {"error": "Metadata parsing failed"}
            except json.JSONDecodeError as e:
                logger.warning(f"[{run_id}] JSONDecodeError parsing metadata: {e}. Response: {metadata_response[:200]}...")
                print(f"Warning: [{run_id}] Could not parse metadata JSON.")
                metadata = {"error": "Metadata JSON parsing failed"}
        else:
            logger.warning(f"[{run_id}] AI failed to generate metadata response.")
            print(f"Warning: [{run_id}] AI failed metadata generation.")
            metadata = {"error": "AI metadata generation failed"}
    else:
        logger.warning(f"[{run_id}] No metadata chunk identified.")
        print(f"Warning: [{run_id}] No metadata chunk found.")
        remaining_text = full_text # Use full text if no metadata chunk

    if not remaining_text or not remaining_text.strip():
        logger.warning(f"[{run_id}] No remaining text after metadata extraction for question parsing.")
        return {"metadata": metadata, "questions": []}

    text_batches = create_text_batches(remaining_text, batch_char_size)
    if not text_batches:
        logger.warning(f"[{run_id}] Failed to create text batches from remaining text.")
        return {"metadata": metadata, "questions": []}

    logger.info(f"[{run_id}] Parsing {len(text_batches)} batches using REAL AI with ENHANCED prompt...")
    all_questions = parse_spg_batches(text_batches, run_id)

    final_consolidated_questions = _consolidate_duplicate_qids(all_questions, f"[Run:{run_id} Final Pass]")
    logger.info(f"[{run_id}] Questions after final consolidation: {len(final_consolidated_questions)}")

    try:
        all_q_sorted = sorted(final_consolidated_questions, key=cmp_to_key(compare_qids))
        logger.info(f"[{run_id}] Sorting complete.")
    except Exception as e:
        logger.error(f"[{run_id}] Error during final sorting: {e}. Returning unsorted.", exc_info=True)
        all_q_sorted = final_consolidated_questions

    try:
        all_q_propagated = _propagate_grouping_instructions(all_q_sorted, run_id)
        logger.info(f"[{run_id}] Instruction propagation complete.")
    except Exception as e:
        logger.error(f"[{run_id}] Error propagating instructions: {e}. Using sorted list.", exc_info=True)
        all_q_propagated = all_q_sorted
    
    try:
        all_questions_final = _parse_raw_hovertexts(all_q_propagated, run_id)
        logger.info(f"[{run_id}] Final Python hovertext parsing complete.")
    except Exception as e:
        logger.error(f"[{run_id}] Error parsing hovertext strings: {e}. Using list before hovertext parsing.", exc_info=True)
        for q_item in all_q_propagated:
            q_item['hovertexts'] = q_item.get('hovertexts', {}) if isinstance(q_item.get('hovertexts'), dict) else {}
        all_questions_final = all_q_propagated

    final_json_output = {"metadata": metadata, "questions": all_questions_final}
    logger.info(f"[{run_id}] REAL SPG parsing process completed with AI-centric logic.")
    return final_json_output

# END OF spg_parser.py FILE