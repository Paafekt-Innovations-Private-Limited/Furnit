*"----------------------------------------------------------------------
*"* Wrapper for legacy FM BAPI_ACC_ACT_POSTINGS_REVERSE
*"* Purpose: reverse FI postings
*"* Modern API: I_JournalEntryTP (RAP BO)
*"----------------------------------------------------------------------
CLASS zcl_sample_bapi_acc_act_postings_reverse_wrapper DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS call
      IMPORTING
        iv_fm_name TYPE string DEFAULT 'BAPI_ACC_ACT_POSTINGS_REVERSE'
      RETURNING
        VALUE(rv_success) TYPE abap_bool.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS zcl_sample_bapi_acc_act_postings_reverse_wrapper IMPLEMENTATION.
  METHOD call.
    CALL FUNCTION iv_fm_name
      EXCEPTIONS
        OTHERS = 1.
    rv_success = xsdbool( sy-subrc = 0 ).
  ENDMETHOD.
ENDCLASS.
