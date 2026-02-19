*"----------------------------------------------------------------------
*"* Wrapper for legacy FM FI_COMPANY_CODE_DATA
*"* Purpose: company code data
*"* Modern API: I_CompanyCode (CDS view)
*"----------------------------------------------------------------------
CLASS zcl_zsample_fi_company_code_data_wrapper DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS call
      IMPORTING
        iv_fm_name TYPE string DEFAULT 'FI_COMPANY_CODE_DATA'
      RETURNING
        VALUE(rv_success) TYPE abap_bool.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS zcl_zsample_fi_company_code_data_wrapper IMPLEMENTATION.
  METHOD call.
    CALL FUNCTION iv_fm_name
      EXCEPTIONS
        OTHERS = 1.
    rv_success = xsdbool( sy-subrc = 0 ).
  ENDMETHOD.
ENDCLASS.
